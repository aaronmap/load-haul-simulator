import simpy
import random
import math
import sys
import pandas as pd
import matplotlib.pyplot as plt

# --- 0. GLOBAL FLAGS ---
DEBUG_MODE = False

# --- 1. CONFIGURATION ---
DAYS_TO_RUN = 14
SHIFTS_PER_DAY = 2
TOTAL_SHIFTS = DAYS_TO_RUN * SHIFTS_PER_DAY
SHIFT_HOURS = 12
SIM_MINUTES = SHIFT_HOURS * 60

# PM Configuration
ANNUAL_HOURS = 365*24 
COST_PER_DOWNTIME_HR = 2*70 
MIN_SMU_HRS = 450000 
MAX_SMU_HRS = 650000 

SHVL_MIN_SMU_HRS = 40000
SHVL_MAX_SMU_HRS = 90000

# SET YOUR TRUCK COUNT HERE
NUM_TRUCKS_TOTAL = 23 
TRUCK_PAYLOAD_MEAN = 200
TRUCK_PAYLOAD_STD = 15
PER_FLEET_NEEDS_FUEL_DURING_SHIFT = 0.25 

# IS THERE A LEADING HAND TO FILL IN FOR THE SHOVEL OPERATOR?
SHOVEL_RELIEF = 11/14 

# GO LINE (SCENARIO B)
GO_LINE_EMPTY = 10.0      # Minutes to travel to Go Line
GO_LINE_EMPTY_STD = 5.0   # Variation

# COSTS
COST_TRUCK_SMU = 210.00   
COST_SHOVEL_SMU = 450.00  

# --- 2. DEFINITIONS ---

class DelayProfile:
    def __init__(self, name, mean, std_dev):
        self.name, self.mean, self.std_dev = name, mean, std_dev
    def get_duration(self):
        return max(0.0, random.gauss(self.mean, self.std_dev))

# description, mean, standard deviation
D_MANUVER = DelayProfile("Manuver",0.30,0.30)
D_LOAD  = DelayProfile("Load", 3.0, 0.5)
D_DUMP  = DelayProfile("Dump", 1.5, 0.2)
D_FUEL  = DelayProfile("Fuel", 20.0,5.0)

D_BLAST = DelayProfile("Blast", 45.0, 10.0)
D_LUNCH = DelayProfile("Lunch",30.0,2.0)
D_BREAK = DelayProfile("Smoko",10.0,3.0)
#Trucks
D_SHIFT_START = DelayProfile("StartShift",20.0,5.0)
D_SHIFT_END = DelayProfile("LastLoad",20.0,5.0)
#Shovels
D_SHIFT_START_SH = DelayProfile("Shovel StartShift", 15.0, 7.0)
D_SHIFT_END_SH = DelayProfile("Shovel EndShift", 15.0,7.0)

# QUEUE BUFFER BUFFER - in minutes
EST_QUEUE_BUFFER = -1

# Lunch Window (Staggered)
LUNCH_WIN_START = 330 # DS 11:30
LUNCH_WIN_END   = 450 # DS 13:30

#Break Window (Staggered for Trucks)
BREAK_WIN_START = 125
BREAK_WIN_END = 215

# Blast Schedule
BLAST_CHANCE = 3/14  
BLAST_START  = 360 

# --- RELIABILITY PROFILES (Mean, Std Dev) ---
PROF_TRUCK_MTBF = DelayProfile("Trk MTBF", 40 * 60, 10 * 60) 
PROF_TRUCK_MTTR = DelayProfile("Trk MTTR", 90, 20)

PROF_SHOVEL_MTBF = DelayProfile("Shv MTBF", 100 * 60, 15 * 60)
PROF_SHOVEL_MTTR = DelayProfile("Shv MTTR", 120, 30)

def generate_lunch_schedule(unit_list, win_start, win_end, duration_profile):
    schedule = {}
    slot_size = duration_profile.mean
    total_window = win_end - win_start
    total_slots = math.floor(total_window / slot_size)
    if total_slots < 1: raise ValueError("Lunch window too short")

    for i, unit in enumerate(unit_list):
        slot_index = i % total_slots
        start_time = win_start + (slot_index * slot_size)
        schedule[unit] = start_time
    return schedule

class ShovelConfig:
    def __init__(self, name, haul_mean, haul_std, return_mean, return_std, priority):
        self.name = name
        self.haul_mean = haul_mean
        self.haul_std = haul_std
        self.return_mean = return_mean
        self.return_std = return_std
        self.priority = priority

# --- DEFINE YOUR SHOVELS HERE ---
SHOVEL_FLEET = [
    ShovelConfig("EX01",   12.0, 2.0, 8.0, 1.2, 1), 
    ShovelConfig("EX02",   22.0, 3.0, 18.0,2.1, 2),
    ShovelConfig("EX03",   30.0, 5.0, 22.0,4.1, 2),
]

# --- 3. SIMULATION AGENTS ---

class ShovelAgent:
    def __init__(self, env, config, stats, is_blast_shift, lunch_start_time, planned_pm_data, is_required=True):
        self.env = env
        self.config = config
        self.global_stats = stats 
        self.res = simpy.PriorityResource(env, capacity=1)
        self.lunch_start_time = lunch_start_time
        self.planned_pm_data = planned_pm_data 
        self.is_required = is_required
        
        # LOGIC FLAGS
        self.accepting_trucks = True  # The "Gate" for new arrivals
        self.breakdown_event = env.event() # Event to notify queued trucks of a failure
        self.current_mttr = 0.0 # The "Shop Announcement" duration
        self.last_break_end_time = 0.0

        # LOCAL STATS 
        self.my_tonnes = 0.0 
        self.my_op = 0.0
        self.my_down = 0.0
        self.my_planned_down = 0.0 
        self.my_parked = 0.0 
        self.my_cal = SIM_MINUTES 
        
        # State Flags
        self.is_broken = False
        self.is_on_break = False 
        self.is_blast_delay = False 
        self.is_blast_shift = is_blast_shift
        self.is_on_pm = False
        self.pm_end_time = 0.0

        # For Debug
        self.state_history =[(0,'shovel_op')]
        self.current_queue_count = 0
        self.queue_history = [(0,0)]

        if not self.is_required:
            self.env.process(self.run_parked_shift())
            return

        # Breakdown Generation
        mtbf_min = PROF_SHOVEL_MTBF.mean
        prob_failure = 1.0 - math.exp(-SIM_MINUTES / mtbf_min)
        if random.random() < prob_failure:
            self.failure_time = random.uniform(0, SIM_MINUTES)
            self.env.process(self.breakdown_trigger())
        
        # Start Processes
        self.env.process(self.shift_logic())
        self.env.process(self.schedule_monitor())
        
        if self.planned_pm_data:
            self.env.process(self.maintenance_trigger())

    @property
    def is_available(self):
        return not (self.is_broken or self.is_on_pm)

    # --- Shift Logic with Dynamic Last Load ---
    def shift_logic(self):
        # 1. Start Shift 
        start_delay = D_SHIFT_START_SH.get_duration()
        with self.res.request(priority=-4) as req:
            yield req
            self.update_debug_status('shovel_parked')
            if start_delay > 0:
                self.my_parked += start_delay
                self.global_stats['shovel_parked'] += start_delay
                yield self.env.timeout(start_delay)
        self.update_debug_status('shovel_op')

        # 2. Monitor for "Dynamic Last Load"
        while True:
            # Dynamic Cutoff: Haul Time + Return to GoLine
            cutoff_duration = self.config.haul_mean + GO_LINE_EMPTY
            time_remaining = SIM_MINUTES - self.env.now
            
            if time_remaining <= cutoff_duration:
                # CLOSE THE GATE
                self.accepting_trucks = False
                # Wait for queue to clear
                while len(self.res.queue) > 0 or len(self.res.users) > 0:
                    yield self.env.timeout(1)
                break
            yield self.env.timeout(1)

        # 3. End of Shift
        with self.res.request(priority=-4) as req:
            yield req
            self.update_debug_status('shovel_parked')
            remaining = SIM_MINUTES - self.env.now
            if remaining > 0:
                self.my_parked += remaining
                self.global_stats['shovel_parked'] += remaining
                yield self.env.timeout(remaining)

    # --- Lunch Monitor with Queue Clearing ---
    def schedule_monitor(self):
        # 1. SMOKO 
        if random.random() > SHOVEL_RELIEF:
            smoko_start = random.uniform(BREAK_WIN_START, BREAK_WIN_END)
            delay = smoko_start - self.env.now
            if delay > 0: yield self.env.timeout(delay)

            with self.res.request(priority=-2) as req:
                yield req
                self.is_on_break = True
                actual_duration = min(D_BREAK.get_duration(), SIM_MINUTES - self.env.now)
                if actual_duration > 0:
                    self.my_parked += actual_duration
                    self.global_stats['shovel_parked'] += actual_duration
                    yield self.env.timeout(actual_duration)
                self.is_on_break = False
                self.last_break_end_time = self.env.now

        # 2. LUNCH (WITH QUEUE CLEARING)
        if self.lunch_start_time > 0:
            delay = self.lunch_start_time - self.env.now
            if delay > 0: yield self.env.timeout(delay)
            
            # Close Gate and Drain Queue
            self.accepting_trucks = False
            while len(self.res.queue) > 0 or len(self.res.users) > 0:
                yield self.env.timeout(1)
            
            with self.res.request(priority=-2) as req:
                yield req
                self.is_on_break = True
                actual_duration = min(D_LUNCH.get_duration(), SIM_MINUTES - self.env.now)
                if actual_duration > 0:
                    self.update_debug_status('shovel_parked')
                    self.my_parked += actual_duration
                    self.global_stats['shovel_parked'] += actual_duration
                    yield self.env.timeout(actual_duration)
                self.update_debug_status('shovel_op')
                self.is_on_break = False
                self.last_break_end_time = self.env.now
            
            self.accepting_trucks = True

        # 3. BLAST
        if self.is_blast_shift:
            time_to_blast = BLAST_START - self.env.now
            if time_to_blast > 0: yield self.env.timeout(time_to_blast)
            with self.res.request(priority=-2) as req:
                yield req
                self.is_blast_delay = True 
                actual_duration = min(D_BLAST.get_duration(), SIM_MINUTES - self.env.now)
                if actual_duration > 0:
                    self.update_debug_status('shovel_parked')
                    self.my_parked += actual_duration
                    self.global_stats['shovel_parked'] += actual_duration
                    yield self.env.timeout(actual_duration)
                self.update_debug_status('shovel_op')
                self.is_blast_delay = False

    # --- Breakdown with Event Trigger ---
    def breakdown_trigger(self):
        yield self.env.timeout(self.failure_time)
        
        self.is_broken = True
        raw_repair_time = PROF_SHOVEL_MTTR.get_duration()
        self.current_mttr = raw_repair_time # Shop Announcement
        
        # Notify trucks in queue
        self.breakdown_event.succeed() 
        self.breakdown_event = self.env.event() # Reset for next time

        with self.res.request(priority=-1) as req:
            yield req
            actual_repair_time = min(raw_repair_time, SIM_MINUTES - self.env.now)
            if actual_repair_time > 0:
                self.update_debug_status('shovel_down')
                self.my_down += actual_repair_time
                self.global_stats['shovel_down'] += actual_repair_time
                yield self.env.timeout(actual_repair_time)

        self.update_debug_status('shovel_op')
        self.is_broken = False
        self.current_mttr = 0.0

    def run_parked_shift(self):
        self.update_debug_status('shovel_parked')
        self.my_parked += SIM_MINUTES
        self.global_stats['shovel_parked'] += SIM_MINUTES
        yield self.env.timeout(SIM_MINUTES)
    def truck_arrived(self):
        if DEBUG_MODE:
            self.current_queue_count += 1
            self.queue_history.append((self.env.now, self.current_queue_count))
    def truck_departed(self):
        if DEBUG_MODE:
            self.current_queue_count = max(0, self.current_queue_count -1)
            self.queue_history.append((self.env.now, self.current_queue_count))
    def update_debug_status(self, new_status):
        if DEBUG_MODE:
            self.state_history.append((self.env.now, new_status))
    def log_op(self, duration):
        self.my_op += duration
        self.global_stats['shovel_op'] += duration 
    def log_tonnes(self,tonnes):
        self.my_tonnes += tonnes
    def maintenance_trigger(self):
        start_time, duration = self.planned_pm_data
        delay = start_time - self.env.now
        if delay > 0: yield self.env.timeout(delay)
        with self.res.request(priority=-3) as req:
            yield req
            self.is_on_pm = True
            actual_loss = min(duration, SIM_MINUTES - self.env.now)
            self.pm_end_time = self.env.now + duration
            if actual_loss > 0:
                self.update_debug_status('shovel_down')
                self.my_planned_down += actual_loss
                self.global_stats['shovel_planned_down'] += actual_loss
                yield self.env.timeout(actual_loss)
            self.is_on_pm = False
            self.update_debug_status('shovel_op')

class TruckAgent:
    def __init__(self, env, truck_id, assigned_shovel_agent, stats):
        self.env = env
        self.id = truck_id
        self.shovel_agent = assigned_shovel_agent
        self.stats = stats
        mtbf_minutes = PROF_TRUCK_MTBF.mean 
        time_to_failure = random.expovariate(1.0 / mtbf_minutes)
        self.next_breakdown = self.env.now + time_to_failure
        self.needs_fuel = (random.random() < PER_FLEET_NEEDS_FUEL_DURING_SHIFT)
        self.has_fueled = False
        self.fuel_trigger_time = random.uniform(120, SIM_MINUTES - 120)
        self.smoko_time = random.uniform(BREAK_WIN_START, BREAK_WIN_END)
        self.taken_smoko = False
        self.env.process(self.cycle())

    def cycle(self):
        start_delay = D_SHIFT_START.get_duration()
        self.stats['truck_standby'] += start_delay
        yield self.env.timeout(start_delay)

        while True:
            # 1. End Shift
            my_cutoff = D_SHIFT_END.get_duration()
            if self.env.now >= (SIM_MINUTES - my_cutoff):
                time_left = SIM_MINUTES - self.env.now
                if time_left > 0:
                    self.stats['truck_parked'] += time_left
                    yield self.env.timeout(time_left)
                break 

            # 2. Breakdowns
            if self.env.now >= self.next_breakdown:
                repair_time = PROF_TRUCK_MTTR.get_duration()
                actual_repair = min(repair_time, SIM_MINUTES - self.env.now)
                if actual_repair > 0:
                    self.stats['truck_down'] += actual_repair
                    yield self.env.timeout(actual_repair)
                mtbf_minutes = PROF_TRUCK_MTBF.mean
                self.next_breakdown = self.env.now + random.expovariate(1.0 / mtbf_minutes)
                continue

            # 3. Smoko
            if not self.taken_smoko and self.env.now >= self.smoko_time:
                break_duration = D_BREAK.get_duration()
                self.stats['truck_standby'] += break_duration
                yield self.env.timeout(break_duration)
                self.taken_smoko = True
                continue

            # 4. Gate Check (Lunch / Last Load / PM)
            if self.shovel_agent.is_on_pm:
                time_to_wait = self.shovel_agent.pm_end_time - self.env.now
                if time_to_wait > 0:
                    self.stats['truck_parked'] += time_to_wait
                    yield self.env.timeout(time_to_wait)
                continue

            if not self.shovel_agent.accepting_trucks:
                # Divert to Go Line
                travel = max(0, random.gauss(GO_LINE_EMPTY, GO_LINE_EMPTY_STD))
                yield self.env.timeout(travel)
                self.stats['truck_op'] += travel
                
                while not self.shovel_agent.accepting_trucks:
                    yield self.env.timeout(5)
                    self.stats['truck_standby'] += 5
                
                travel_back = max(0, random.gauss(GO_LINE_EMPTY, GO_LINE_EMPTY_STD))
                yield self.env.timeout(travel_back)
                self.stats['truck_op'] += travel_back
                continue

            # 5. ARRIVE AND QUEUE (With Reneging)
            arrive_time = self.env.now
            self.shovel_agent.truck_arrived()
            req = self.shovel_agent.res.request(priority=0)
            joined_queue = True
            
            try:
                while True:
                    # Wait for Resource OR Breakdown
                    results = yield req | self.shovel_agent.breakdown_event
                    
                    if req in results:
                        break # Got it!
                    else:
                        # Shovel Broke while we were queuing
                        yield self.env.timeout(10) # 10 min "Wait and See"
                        self.stats['truck_standby'] += 10
                        
                        # Shop Announcement Check
                        threshold = 2 * GO_LINE_EMPTY
                        if self.shovel_agent.current_mttr > threshold:
                            # RENEGE
                            req.cancel()
                            self.shovel_agent.truck_departed()
                            joined_queue = False
                            
                            # Drive to Go Line
                            travel = max(0, random.gauss(GO_LINE_EMPTY, GO_LINE_EMPTY_STD))
                            yield self.env.timeout(travel)
                            self.stats['truck_op'] += travel
                            
                            # Wait Loop
                            while not self.shovel_agent.is_available:
                                yield self.env.timeout(5)
                                self.stats['truck_standby'] += 5
                            
                            # Return
                            travel_back = max(0, random.gauss(GO_LINE_EMPTY, GO_LINE_EMPTY_STD))
                            yield self.env.timeout(travel_back)
                            self.stats['truck_op'] += travel_back
                            break 
                        else:
                            pass # Stay in queue
            except simpy.Interrupt:
                req.cancel()
                joined_queue = False

            if not joined_queue: continue

            # 6. LOADING
            current_time = self.env.now
            total_wait_time = current_time - arrive_time
            delay_deduction = 0.0
            if self.shovel_agent.last_break_end_time > arrive_time:
                overlap = self.shovel_agent.last_break_end_time - arrive_time
                delay_deduction = min(total_wait_time, overlap)
            
            if delay_deduction > 0: self.stats['truck_standby'] += delay_deduction
            self.stats['truck_queue'] += (total_wait_time - delay_deduction)

            maneuver_time = D_MANUVER.get_duration()
            yield self.env.timeout(maneuver_time)
            self.stats['truck_op'] += maneuver_time
            self.shovel_agent.log_op(maneuver_time)

            load_time = D_LOAD.get_duration()
            yield self.env.timeout(load_time)
            self.stats['truck_op'] += load_time
            self.shovel_agent.log_op(load_time) 

            self.shovel_agent.truck_departed()
            self.shovel_agent.res.release(req) 

            payload = random.gauss(TRUCK_PAYLOAD_MEAN, TRUCK_PAYLOAD_STD)
            self.stats['tonnes'] += payload
            self.stats['loads'] += 1
            self.shovel_agent.log_tonnes(payload)

            # 7. HAUL
            haul_time = max(0, random.gauss(self.shovel_agent.config.haul_mean, self.shovel_agent.config.haul_std))
            yield self.env.timeout(haul_time)
            self.stats['truck_op'] += haul_time
            
            # 8. DUMP
            dump_time = D_DUMP.get_duration()
            yield self.env.timeout(dump_time)
            self.stats['truck_op'] += dump_time

            # 9. FUEL
            if self.needs_fuel and not self.has_fueled and self.env.now >= self.fuel_trigger_time:
                fuel_time = D_FUEL.get_duration()
                self.stats['truck_standby'] += fuel_time
                yield self.env.timeout(fuel_time) 
                self.has_fueled = True
            
            # 10. RETURN (FORK)
            # If shovel is down OR gate is closed, go to GoLine
            if not self.shovel_agent.is_available or not self.shovel_agent.accepting_trucks:
                travel_to_goline = max(0, random.gauss(GO_LINE_EMPTY, GO_LINE_EMPTY_STD))
                yield self.env.timeout(travel_to_goline)
                self.stats['truck_op'] += travel_to_goline 

                while not self.shovel_agent.is_available or not self.shovel_agent.accepting_trucks:
                    yield self.env.timeout(5)
                    self.stats['truck_standby'] += 5 

                travel_from_goline = max(0, random.gauss(GO_LINE_EMPTY, GO_LINE_EMPTY_STD))
                yield self.env.timeout(travel_from_goline)
                self.stats['truck_op'] += travel_from_goline
            else:
                return_time = max(0, random.gauss(self.shovel_agent.config.return_mean, self.shovel_agent.config.return_std))
                yield self.env.timeout(return_time)
                self.stats['truck_op'] += return_time

# --- 4. REPORTING & DISPATCH (Same as before) ---
def print_shift_header(data):
    header = (f"{'Day':<4} {'Shift':<6} {'Avail':<6} {'Tonnes':<8} "
              f"{'TrkAv%':<7} {'TrkUt%':<7} {'TrkEf%':<7} {'Q/Trk':<7} "
              f"{'ShvAv':<6} {'ShvAv%':<7} {'ShvUt%':<7} {'ShvEf%':<7} {'Hang/Shv':<9}")
    print("-" * len(header)); print(header); print("-" * len(header))

def print_shift_row(s, data):
    t_cal = NUM_TRUCKS_TOTAL * SIM_MINUTES
    t_down_total = data['truck_down'] + data['planned_down']
    t_avail_time = t_cal - t_down_total
    avg_trucks_avail = t_avail_time / SIM_MINUTES
    t_smu = data['truck_op'] + data['truck_queue'] + data['truck_standby']
    t_operating = data['truck_op'] + data['truck_queue']
    trk_avail_pct = (t_avail_time / t_cal) * 100
    trk_util_pct  = (t_smu / t_avail_time) * 100 if t_avail_time > 0 else 0
    trk_eff_pct   = (t_operating / t_smu) * 100 if t_smu > 0 else 0
    q_per_truck   = data['truck_queue'] / avg_trucks_avail if avg_trucks_avail > 0 else 0

    num_shovels = len(SHOVEL_FLEET)
    s_cal = num_shovels * SIM_MINUTES
    s_down_total = data['shovel_down'] + data['shovel_planned_down']
    s_avail_time = s_cal - s_down_total
    avg_shovels_avail = s_avail_time / SIM_MINUTES
    total_hang = sum(data['hang_breakdown'].values())
    s_utilized_time = data['shovel_op'] + total_hang
    shv_avail_pct = (s_avail_time / s_cal) * 100
    shv_util_pct  = (s_utilized_time / s_avail_time) * 100 if s_avail_time > 0 else 0
    shv_eff_pct   = (data['shovel_op'] / s_utilized_time) * 100 if s_utilized_time > 0 else 0
    hang_per_shovel = total_hang / avg_shovels_avail if avg_shovels_avail > 0 else 0

    day = (s // 2) + 1; shft = "D" if s % 2 == 0 else "N"
    print(f"{day:<4} {shft:<6} {avg_trucks_avail:<6.1f} {data['tonnes']:<8,.0f} "
          f"{trk_avail_pct:<7.1f} {trk_util_pct:<7.1f} {trk_eff_pct:<7.1f} {q_per_truck:<7.1f} "
          f"{avg_shovels_avail:<6.1f} {shv_avail_pct:<7.1f} {shv_util_pct:<7.1f} {shv_eff_pct:<7.1f} {hang_per_shovel:<9.0f}")

def print_final_report(shift_data_list, df_pm, planned_down_hours):
    agg = {k: sum(d[k] for d in shift_data_list) for k in shift_data_list[0] if k in ['tonnes', 'loads', 'truck_op', 'truck_queue', 'truck_standby', 'truck_parked', 'truck_down', 'shovel_op', 'shovel_cal', 'shovel_down', 'shovel_parked']}  
    hrs_active, hrs_queue = agg['truck_op']/60, agg['truck_queue']/60
    hrs_standby, hrs_parked = agg['truck_standby']/60, agg['truck_parked']/60
    hrs_operating = hrs_active + hrs_queue
    hrs_smu = hrs_operating + hrs_standby
    hrs_cal = (NUM_TRUCKS_TOTAL * TOTAL_SHIFTS * SHIFT_HOURS)
    hrs_down= (agg['truck_down']/60) + planned_down_hours
    hrs_avail = hrs_cal - hrs_down
    shovel_smu_hrs = (agg['shovel_op'] / 60) + max(0, (agg['shovel_cal']/60) - (agg['shovel_down']/60) - (agg['shovel_parked']/60) - (agg['shovel_op']/60))
    monthly_mult = 30.4 / DAYS_TO_RUN
    
    print("\n" + "="*80 + "\n PERFORMANCE SUMMARY (FLEET)\n" + "="*80)    
    summary_data = [
        {"Metric": "Total Tonnes",     "Unit": "t",       "Sim": agg['tonnes'],  "Monthly": agg['tonnes']*monthly_mult},
        {"Metric": "Operating Hrs",    "Unit": "hrs",     "Sim": hrs_operating,  "Monthly": hrs_operating*monthly_mult},
        {"Metric": "  - Active",       "Unit": "hrs",     "Sim": hrs_active,      "Monthly": "-"},
        {"Metric": "  - Queue",        "Unit": "hrs",     "Sim": hrs_queue,       "Monthly": "-"},
        {"Metric": "Standby Hrs",      "Unit": "hrs",     "Sim": hrs_standby,     "Monthly": "-"},
        {"Metric": "Parked Hrs",       "Unit": "hrs",     "Sim": hrs_parked,      "Monthly": "-"},
        {"Metric": "Truck Avail",      "Unit": "%",       "Sim": (hrs_avail/hrs_cal)*100, "Monthly": "-"},
        {"Metric": "Asset Util",       "Unit": "%",       "Sim": (hrs_smu/hrs_avail)*100, "Monthly": "-"},
        {"Metric": "Effective Util",   "Unit": "%",       "Sim": (hrs_active/hrs_operating)*100, "Monthly": "-"},
        {"Metric": "Fleet Prod",       "Unit": "t/OpHr",  "Sim": agg['tonnes']/hrs_operating, "Monthly": "-"},
    ]    
    df_sum = pd.DataFrame(summary_data)
    pd.options.display.float_format = '{:,.1f}'.format
    print(df_sum.to_string(index=False))

    print("\n" + "="*80 + "\n SHOVEL PERFORMANCE REPORT\n" + "="*80)
    shovel_totals = {}
    for d in shift_data_list:
        for name, stats in d['shovel_perf'].items():
            if name not in shovel_totals:
                shovel_totals[name] = {'tonnes': 0, 'down': 0, 'planned_down': 0, 'op': 0, 'hang': 0, 'cal': 0}
            for k in shovel_totals[name]: shovel_totals[name][k] += stats.get(k, 0)
            shovel_totals[name]['cal'] += SIM_MINUTES

    shv_report_data = []
    for name, s in shovel_totals.items():
        cal_hrs = s['cal'] / 60
        down_hrs = (s['down'] + s['planned_down']) / 60
        op_hrs, hang_hrs = s['op'] / 60, s['hang'] / 60
        avail_hrs = cal_hrs - down_hrs
        user_op_hours = op_hrs + hang_hrs
        
        shv_report_data.append({
            "Shovel ID": name,
            "Tonnes": s['tonnes'],
            "Avail %": ((avail_hrs / cal_hrs) * 100) if cal_hrs > 0 else 0,
            "Util %": ((user_op_hours / avail_hrs) * 100) if avail_hrs > 0 else 0,
            "Op Hrs": user_op_hours,
            "Prod (t/hr)": s['tonnes'] / user_op_hours if user_op_hours > 0 else 0,
            "Hang Hrs": hang_hrs
        })
    print(pd.DataFrame(shv_report_data).to_string(index=False))

    print("\n" + "-"*80 + "\n COST PROJECTION\n" + "-"*80)
    cost_data = []
    for name, smu, rate in [("Shovel Fleet", shovel_smu_hrs, COST_SHOVEL_SMU), ("Truck Fleet",  hrs_smu, COST_TRUCK_SMU)]:
        total_sim = smu * rate
        cost_data.append({
            "Fleet": name, "$/SMU Rate": rate, "Total $ (Sim)": total_sim,
            "$/t (Unit Cost)": f"{total_sim / agg['tonnes']:.2f}"
        })
    print(pd.DataFrame(cost_data).to_string(index=False))
    print("\n" + "="*80 + "\n")

def run_shift(shift_num, pm_schedule):
    env = simpy.Environment()
    stats = {'tonnes': 0, 'loads': 0, 'truck_op': 0.0, 'truck_queue': 0.0, 'truck_standby': 0.0, 'truck_parked': 0.0, 
             'truck_down': 0.0, 'planned_down': 0.0, 'shovel_op': 0.0, 'shovel_down': 0.0, 'shovel_planned_down': 0.0, 
             'shovel_parked': 0.0, 'shovel_cal': 0.0, 'dispatch_log': [], 'hang_breakdown': {}, 'shovel_perf': {}, 'planned_down_ids': []}    
    
    is_blast = (random.random() < BLAST_CHANCE)
    shift_start_hr = shift_num * SHIFT_HOURS
    shift_end_hr = (shift_num + 1) * SHIFT_HOURS

    trucks_down_for_pm = set()
    if not pm_schedule.empty:
        relevant_truck_pms = pm_schedule[(pm_schedule["Asset Type"] == "TRUCK") & (pm_schedule["Hours_Into_Sim"] < shift_end_hr) & ((pm_schedule["Hours_Into_Sim"] + pm_schedule["Downtime"]) > shift_start_hr)]
        for _, row in relevant_truck_pms.iterrows():
            trucks_down_for_pm.add(row["Asset ID"]); stats['planned_down_ids'].append(f"{row['Asset ID']} ({row['Event']})") 

    shovel_pm_map = {} 
    if not pm_schedule.empty:
        relevant_shovel_pms = pm_schedule[(pm_schedule["Asset Type"] == "SHOVEL") & (pm_schedule["Hours_Into_Sim"] < shift_end_hr) & ((pm_schedule["Hours_Into_Sim"] + pm_schedule["Downtime"]) > shift_start_hr)]
        for _, row in relevant_shovel_pms.iterrows():
            start_relative = max(0, row["Hours_Into_Sim"] - shift_start_hr) * 60
            shovel_pm_map[row["Asset ID"]] = (start_relative, row["Downtime"] * 60)

    available_truck_pool = []
    for i in range(NUM_TRUCKS_TOTAL):
        t_id = f"T-{i}" 
        if t_id not in trucks_down_for_pm: available_truck_pool.append(t_id)
        else: stats['planned_down'] += SIM_MINUTES 

    remaining_trucks = len(available_truck_pool)
    active_configs = sorted(SHOVEL_FLEET, key=lambda x: x.priority)
    shovel_lunch_times = generate_lunch_schedule(active_configs, LUNCH_WIN_START, LUNCH_WIN_END, D_LUNCH)
    created_agents = {}

    for cfg in active_configs:
        cycle_time = (D_MANUVER.mean + D_LOAD.mean + cfg.haul_mean + D_DUMP.mean + cfg.return_mean + EST_QUEUE_BUFFER)
        trucks_needed = math.ceil(cycle_time / (D_LOAD.mean + D_MANUVER.mean))
        assigned_count = min(remaining_trucks, trucks_needed)
        is_req = (assigned_count>0)
        
        agent = ShovelAgent(env, cfg, stats, is_blast, shovel_lunch_times[cfg], shovel_pm_map.get(cfg.name, None), is_required=is_req)
        created_agents[cfg.name] = agent; stats['shovel_cal'] += SIM_MINUTES 

        if is_req:
            for _ in range(assigned_count): TruckAgent(env, available_truck_pool.pop(0), agent, stats)
            remaining_trucks -= assigned_count

    if len(available_truck_pool) > 0: stats['truck_parked'] += (len(available_truck_pool) * SIM_MINUTES)
    env.run(until=SIM_MINUTES)
    if DEBUG_MODE: generate_shovel_debug_graph(created_agents, shift_num)    
    
    for cfg in SHOVEL_FLEET:
        if cfg.name in created_agents:
            agent = created_agents[cfg.name]
            hang = max(0, agent.my_cal - agent.my_down - agent.my_planned_down - agent.my_parked - agent.my_op)
            stats['hang_breakdown'][cfg.name] = hang
            stats['shovel_perf'][cfg.name] = {'tonnes': agent.my_tonnes, 'op': agent.my_op, 'down': agent.my_down, 'planned_down': agent.my_planned_down, 'hang': hang, 'parked': agent.my_parked}
        else:
            stats['hang_breakdown'][cfg.name] = 0.0
            stats['shovel_perf'][cfg.name] = {'tonnes': 0, 'op': 0, 'down': 0, 'planned_down': 0, 'hang': 0, 'parked': SIM_MINUTES}
    return stats

def generate_fleet_schedule(num_trucks, sim_duration_hours):
    all_events = []
    print(f"\n=== GENERATING PM PLAN (Scope: {sim_duration_hours} Hours) ===")
    for i in range(num_trucks): 
        truck_id = f"T-{i}"; start_hours = random.randint(MIN_SMU_HRS, MAX_SMU_HRS)
        current_pm_target = ((start_hours // 250) + 1) * 250
        while current_pm_target <= (start_hours + sim_duration_hours):
            if current_pm_target % 1000 == 0: pm_type = "C-Check (12h)"; duration = 12
            elif current_pm_target % 500 == 0: pm_type = "B-Check (8h)"; duration = 8
            else: pm_type = "A-Check (4h)"; duration = 4
            all_events.append({"Asset ID": truck_id, "Asset Type": "TRUCK", "Start Hours": start_hours, "PM Interval": current_pm_target, "Event": pm_type, "Downtime": duration})
            current_pm_target += 250
    
    for shovel in SHOVEL_FLEET:
        start_hours = random.randint(SHVL_MIN_SMU_HRS,SHVL_MAX_SMU_HRS)
        current_pm_target = ((start_hours // 250) + 1) * 250
        while current_pm_target <= (start_hours + sim_duration_hours):
            if current_pm_target % 1000 == 0: pm_type = "C-Check (12h)"; duration = 12
            elif current_pm_target % 500 == 0: pm_type = "B-Check (8h)"; duration = 8
            else: pm_type = "A-Check (4h)"; duration = 4
            all_events.append({"Asset ID": shovel.name, "Asset Type": "SHOVEL", "Start Hours": start_hours, "PM Interval": current_pm_target, "Event": pm_type, "Downtime": duration})
            current_pm_target += 250

    if not all_events: return pd.DataFrame(columns=["Asset ID", "Asset Type", "Start Hours", "PM Interval", "Event", "Downtime"])
    return pd.DataFrame(all_events)

def generate_shovel_debug_graph(shovels_dict, shift_num):
    status_map = {'shovel_parked': 1, 'shovel_down': 2, 'shovel_op': 3}
    num_shovels = len(shovels_dict)
    fig, axs = plt.subplots(num_shovels, 1, figsize=(12, 4*num_shovels), sharex=True)
    if num_shovels == 1: axs = [axs]
    for i, (shovel_name, shovel_agent) in enumerate(shovels_dict.items()):
        times_s, statuses = [], []
        for entry in shovel_agent.state_history + [(SIM_MINUTES, shovel_agent.state_history[-1][1])]:
            times_s.append(entry[0]); statuses.append(status_map.get(entry[1], 3))
        ax1 = axs[i]; ax1.step(times_s, statuses, where='post', color='red', linewidth=2, label='Status')
        ax1.set_yticks([1, 2, 3]); ax1.set_yticklabels(['Parked', 'Down', 'Op']); ax1.set_title(f"Shovel Status: {shovel_name}")
        ax2 = ax1.twinx(); times_q, counts = [], []
        for entry in shovel_agent.queue_history + [(SIM_MINUTES, shovel_agent.queue_history[-1][1])]:
            times_q.append(entry[0]); counts.append(entry[1])
        ax2.step(times_q, counts, where='post', color='blue', linewidth=1.5, alpha=0.7); ax2.set_ylabel("Trucks @ Shovel")
    plt.tight_layout(); plt.savefig(f"shift_{shift_num:03d}_debug.png"); plt.close(fig)

def main():
    sim_duration = TOTAL_SHIFTS * SHIFT_HOURS
    print(f"=== Simulation Setup: {TOTAL_SHIFTS} Shifts ({sim_duration} Hours) ===")
    df_pm = generate_fleet_schedule(NUM_TRUCKS_TOTAL, sim_duration)
    if not df_pm.empty:
        df_pm["Hours_Into_Sim"] = df_pm["PM Interval"] - df_pm["Start Hours"]
        df_pm["Planned Shift"] = (df_pm["Hours_Into_Sim"] // SHIFT_HOURS).astype(int) + 1
        planned_down_hours = df_pm["Downtime"].sum()
    else: df_pm["Hours_Into_Sim"] = 0; planned_down_hours = 0.0

    shift_data_list = []
    print(f"\n=== STARTING SIMULATION FOR {TOTAL_SHIFTS} SHIFTS ===")
    for s in range(TOTAL_SHIFTS):
        data = run_shift(s, df_pm); shift_data_list.append(data)
        if s == 0: print_shift_header(data)
        print_shift_row(s, data)
    print_final_report(shift_data_list, df_pm, planned_down_hours)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "go":
        if len(sys.argv) > 2 and sys.argv[2] =="1": DEBUG_MODE = True
        main()
    else: print("\nRun with: python LHSimulatorDispatch.py go")
