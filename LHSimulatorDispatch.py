import simpy
import random
import math
import sys
import pandas as pd
import matplotlib.pyplot as plt

DEBUG_MODE = False

# --- 1. CONFIGURATION ---
DAYS_TO_RUN = 14
SHIFTS_PER_DAY = 2
TOTAL_SHIFTS = DAYS_TO_RUN * SHIFTS_PER_DAY
SHIFT_HOURS = 12
SIM_MINUTES = SHIFT_HOURS * 60

# -- the only haul cycle time outside of the digger - travel to/from go line
GO_LINE_EMPTY = 10 #minutes to the goline empty
GO_LINE_EMPTY_STD = 5 #a "big" standard deviation to deal with a varity of distances... there is a check to make sure it isn't less than zero


# PM Configuration
ANNUAL_HOURS = 365*24 # the maximum number of hours a year that a truck could run.
COST_PER_DOWNTIME_HR = 2*70 #maintenance manpower costs
MIN_SMU_HRS = 450000 #lowest hour meter reading
MAX_SMU_HRS = 650000 #highest hour meter reading

SHVL_MIN_SMU_HRS = 40000
SHVL_MAX_SMU_HRS = 90000


# SET YOUR TRUCK COUNT HERE
NUM_TRUCKS_TOTAL = 23 
TRUCK_PAYLOAD_MEAN = 200
TRUCK_PAYLOAD_STD = 15
PER_FLEET_NEEDS_FUEL_DURING_SHIFT = 0.25 #as opposed to being fueled during shift change

# IS THERE A LEADING HAND TO FILL IN FOR THE SHOVEL OPERATOR?
# set the number of shift where there is leading hand to
# run the shovel when the operator takes a break.
SHOVEL_RELIEF = 11/14 #means that 11  shifts out of 14 will  have a leading hand to take over the shovel while the operator is on break.

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
# insurance - artifically inflats the cycle time to ensure the schedule is overtrucked
# if negative, has the impact of undertrucking the shovel
EST_QUEUE_BUFFER = -1

# Lunch Window (Staggered)
LUNCH_WIN_START = 330 # DS 11:30
LUNCH_WIN_END   = 450 # DS 13:30

#Break Window (Staggered for Trucks)
BREAK_WIN_START = 125
BREAK_WIN_END = 215

# Blast Schedule
BLAST_CHANCE = 3/14  #probability that there will be a blast
BLAST_START  = 360 #minutes into the shift that the blast occurs 

# --- RELIABILITY PROFILES (Mean, Std Dev) ---
# Trucks: Fail every ~40 hours with a standard deviation of 10 hours
# Trucks have a mean time to return of 1.5 hours with a standard deviation of 20 minutes
PROF_TRUCK_MTBF = DelayProfile("Trk MTBF", 40 * 60, 10 * 60) 
PROF_TRUCK_MTTR = DelayProfile("Trk MTTR", 90, 20)
# Shovels: Fail every ~100 hours, +/- 15 hours
# shoves take 2 hours to return +/- 30 mintes
PROF_SHOVEL_MTBF = DelayProfile("Shv MTBF", 100 * 60, 15 * 60)
PROF_SHOVEL_MTTR = DelayProfile("Shv MTTR", 120, 30)

def generate_lunch_schedule(unit_list, win_start, win_end, duration_profile):
    """
    Distributes units into 30-minute slots within the window.
    Returns a dictionary: { 'Unit_ID': start_time }
    """
    schedule = {}
    slot_size = duration_profile.mean
    total_window = win_end - win_start
    
    # Calculate how many 30-min slots fit in the window (e.g., 120 / 30 = 4 slots)
    total_slots = math.floor(total_window / slot_size)
    
    if total_slots < 1:
        raise ValueError("Lunch window is too short for the break duration!")

    for i, unit in enumerate(unit_list):
        # Round Robin: Cycle through available slots (0, 1, 2, 3, 0, 1...)
        slot_index = i % total_slots
        
        # Calculate start time
        start_time = win_start + (slot_index * slot_size)
        schedule[unit] = start_time
        
    return schedule

class ShovelConfig:
    def __init__(self, name, haul_mean, haul_std, return_mean, return_std, priority):
        self.name = name
        self.haul_mean = haul_mean
        self.haul_std = haul_std
        # New attributes for the return leg
        self.return_mean = return_mean
        self.return_std = return_std
        self.priority = priority

# --- DEFINE YOUR SHOVELS HERE ---
#shovel, trk haul time, haul stdev, trk return time, return dev, priority
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
        self.last_break_end_time = 0.0
        self.is_required = is_required

        #logic flags
        self.accepting_trucks = True        #shovel will accept / not accept trucks returning to the digger
        self.breakdown_event = env.event()  #announce to trucks in the queue if there is a a shovel breakdown.
        self.current_mttr = 0.0             #used for the workshop announcement for the breakdown time
        
        # LOCAL STATS 
        self.my_tonnes = 0.0 #track to shovel
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

        # --- PROBABILISTIC BREAKDOWN (Unplanned) ---
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

    def breakdown_trigger(self):
        yield self.env.timeout(self.failure_time)
        self.is_broken = True
        raw_repair_time = PROF_SHOVEL_MTTR.get.guration()
        self.current_mttr = raw_repair_time

        #fire event to wake up queued trucks
        self.breakdown_event.succeed()
        self.breakdown_event = self.env.event() #reset event 
        with self.res.request(priority=-1) as req:
            yield req
            actual_repair_time = min(raw_repair_time,SIM_MINUTES-self.env.now)
            if actual_repair_time > 0:
                self.update_debug-status('shovel_down')
                self.my_down += actual_repair_time
                self.global_stats['shovel_down'] += actual_repair_time
                yield self.env.timeout(actual_repair_time)
        self.update_debug_status('shovel_op')
        self.is_broken = False
        self.current_mttr = 0.0


    def shift_logic(self):
        """
        Handles the Operator Arrival (Start) and Cleanup/Travel (End).
        Uses Priority -4 (Highest) to block everything else.
        """
        # --- 1. START OF SHIFT DELAY ---
        start_delay = D_SHIFT_START_SH.get_duration()
        
        # Lock resource immediately
        with self.res.request(priority=-4) as req:
            yield req
            self.update_debug_status('shovel_parked')
            
            if start_delay > 0:
                self.my_parked += start_delay
                self.global_stats['shovel_parked'] += start_delay
                yield self.env.timeout(start_delay)

        # Release lock -> Now Operating
        self.update_debug_status('shovel_op')

        #2 Monitor for 'dynmic last load'
        while True:
            #dynamic cutoff - haul time + return to goline
            cutoff_duration = self.config.haul_mean +GO_LINE_EMPTY
            time_remaining = SIM_MINUTES - self.env.now

            if time_remaining <= cutoff_duration:
                self.accepting_trucks = False
                while len(self.res.queue) > 0 or len(self.res.users) > 0:
                    yield self.env.timeout(1)

                #queue clearned, not shutdown
                break
            yield self.env.timeout(1) #check again in a minute

        #3 End of Shift
        with self.res.request(priority=-4) as req:
            yield req
            self.update_debug_status('shovel_parked')
            remaining = SIM_MINUTES - self.env.now
            if remaining > 0:
                self.my_parked += remaining
                self.global_stats['shovel_parked'] += remaining
                yield self.env.timeout(remaining)


    def schedule_monitor(self):
        # 1. SMOKO 
        if random.random() > SHOVEL_RELIEF:
            smoko_start = random.uniform(BREAK_WIN_START, BREAK_WIN_END)
            # Wait for start time
            delay = smoko_start - self.env.now
            if delay > 0:
                yield self.env.timeout(delay)

            # Request Control
            with self.res.request(priority=-2) as req:
                yield req
                self.is_on_break = True
                
                # Clip duration to remaining shift time
                raw_duration = D_BREAK.get_duration()
                actual_duration = min(raw_duration, SIM_MINUTES - self.env.now)
                
                if actual_duration > 0:
                    self.my_parked += actual_duration
                    self.global_stats['shovel_parked'] += actual_duration
                    yield self.env.timeout(actual_duration)

                self.is_on_break = False
                self.last_break_end_time = self.env.now

        # 2. LUNCH
        if self.lunch_start_time > 0:
            delay = self.lunch_start_time - self.env.now
            if delay > 0:
                yield self.env.timeout(delay)

            #1 close the gate...
            self.accepting_trucks = False

            #2 wait for the queue to clear (process existing trucks)...
            while len(self.res.queue) > 0 or len(self.res.users) > 0:
                yield self.env.timeout(1)

            #3 start lunch
            with self.res.request(priority=-2) as req:
                yield req
                self.is_on_break = True
                
                #Clip duration
                raw_duration = D_LUNCH.get_duration()
                actual_duration = min(raw_duration, SIM_MINUTES - self.env.now)
                
                if actual_duration > 0:
                    self.update_debug_status('shovel_parked')
                    self.my_parked += actual_duration
                    self.global_stats['shovel_parked'] += actual_duration
                    yield self.env.timeout(actual_duration)
                    
                self.update_debug_status('shovel_op')
                self.is_on_break = False
                self.last_break_end_time = self.env.now

            #4 re-open the 'gate'
            self.accepting_trucks = True


        # 3. BLAST
        if self.is_blast_shift:
            time_to_blast = BLAST_START - self.env.now
            if time_to_blast > 0:
                yield self.env.timeout(time_to_blast)
            
            with self.res.request(priority=-2) as req:
                yield req
                self.is_blast_delay = True 
                
                # Clip duration
                raw_duration = D_BLAST.get_duration()
                actual_duration = min(raw_duration, SIM_MINUTES - self.env.now)

                if actual_duration > 0:
                    self.update_debug_status('shovel_parked')
                    self.my_parked += actual_duration
                    self.global_stats['shovel_parked'] += actual_duration
                    yield self.env.timeout(actual_duration)

                self.update_debug_status('shovel_op')
                self.is_blast_delay = False

    def breakdown_trigger(self):
        """
        Independent process that triggers Shovel breakdowns.
        """
        yield self.env.timeout(self.failure_time)

        #calculate MTTR...shop announcment on the lenght of time the digger will be down...
        raw_repair_time = PROF_SHOVEL_MTTR.get_duration()
        self.current_mttr = raw_repair_time #for the trucks to see...

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
            self.current_quddeue_count = max(0, self.current_queue_count -1)
            self.queue_history.append((self.env.now, self.current_queue_count))

    def update_debug_status(self, new_status):
        if DEBUG_MODE:
            self.state_history.append((self.env.now, new_status))

    def log_op(self, duration):
        """Called by Truck to log operating time"""
        self.my_op += duration
        self.global_stats['shovel_op'] += duration 

    def log_tonnes(self,tonnes):
        self.my_tonnes += tonnes

    def maintenance_trigger(self):
        """
        Handles PLANNED maintenance (A/B/C Checks).
        """
        start_time, duration = self.planned_pm_data
        
        # 1. Wait for the PM start time
        delay = start_time - self.env.now
        if delay > 0:
            yield self.env.timeout(delay)
            
        # 2. Take Resource (Highest Priority -3)
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

        # Setup Decisions
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
        # 1. Start Delay (Standby)
        start_delay = D_SHIFT_START.get_duration()
        self.stats['truck_standby'] += start_delay
        yield self.env.timeout(start_delay)

        while True:
            # 2. End Shift (Parked)
            my_cutoff = D_SHIFT_END.get_duration()
            if self.env.now >= (SIM_MINUTES - my_cutoff):
                time_left = SIM_MINUTES - self.env.now
                if time_left > 0:
                    self.stats['truck_parked'] += time_left
                    yield self.env.timeout(time_left)
                break 

            # 3. Breakdowns (Down)
            if self.env.now >= self.next_breakdown:
                repair_time = PROF_TRUCK_MTTR.get_duration()
                actual_repair = min(repair_time, SIM_MINUTES - self.env.now)
                if actual_repair > 0:
                    self.stats['truck_down'] += actual_repair
                    yield self.env.timeout(actual_repair)
                
                mtbf_minutes = PROF_TRUCK_MTBF.mean
                next_time = random.expovariate(1.0 / mtbf_minutes)
                self.next_breakdown = self.env.now + next_time
                continue

            # 4. Smoko (Standby)
            if not self.taken_smoko and self.env.now >= self.smoko_time:
                break_duration = D_BREAK.get_duration()
                self.stats['truck_standby'] += break_duration
                yield self.env.timeout(break_duration)
                self.taken_smoko = True
                continue

            # 5. Shovel Status Check
            if self.shovel_agent.is_on_pm:
                time_to_wait = self.shovel_agent.pm_end_time - self.env.now
                actual_wait = min(time_to_wait, SIM_MINUTES - self.env.now)
                if actual_wait > 0:
                    self.stats['truck_parked'] += actual_wait
                    yield self.env.timeout(actual_wait)
                continue

            #check for digger lunch/last load
            if not self.shovel_agent.accepting_trucks:
                travel = max(0,random.gauss(GO_LINE_EMPTY,GO_LINE_EMPTY_STD))
                yield self.env.timeout(travel)
                self.stats['truck_op'] += travel

                #wait until accepting again
                while not self.shovel_agent.accepting_trucks:
                    yield self.env.timeout(5)
                    self.stats['truck_standby'] += 5

                #return
                travel_back = max(0,random.gauss(GO_LINE_EMPTY,GO_LINE_EMPTY_STD))
                yield self.env.timeout(travel_back)
                self.stats['truck_op'] += travel_back
                continue
            
# 4. ARRIVE AND QUEUE (With Reneging Logic)
            arrive_time = self.env.now
            self.shovel_agent.truck_arrived()
            
            # We create the request
            req = self.shovel_agent.res.request(priority=0)
            
            # We wait for EITHER the request to be granted OR the shovel to break
            # This handles trucks ALREADY in queue when shovel breaks
            joined_queue = True
            
            try:
                # Loop allows us to handle multiple "Breakdown -> Wait -> Stay" cycles
                while True:
                    # Wait for Request OR Breakdown Event
                    results = yield req | self.shovel_agent.breakdown_event
                    
                    if req in results:
                        # WE GOT THE SHOVEL!
                        break
                    else:
                        # SHOVEL BROKE WHILE WE WERE IN QUEUE
                        # 1. Wait 10 mins (Standby)
                        yield self.env.timeout(10)
                        self.stats['truck_standby'] += 10
                        
                        # 2. Check Shop Announcement (MTTR)
                        # Rule: If MTTR > 2x Travel Time, we leave.
                        # (Travel Time = GO_LINE_EMPTY = 10. So threshold is 20)
                        threshold = 2 * GO_LINE_EMPTY
                        
                        if self.shovel_agent.current_mttr > threshold:
                            # RENEGE (Leave Queue)
                            req.cancel()
                            self.shovel_agent.truck_departed()
                            joined_queue = False
                            
                            print(f"{self.env.now:.1f}: {self.id} leaving queue due to long breakdown ({self.shovel_agent.current_mttr:.0f}m)")
                            
                            # Drive to Go Line
                            travel = max(0, random.gauss(GO_LINE_EMPTY, GO_LINE_EMPTY_STD))
                            yield self.env.timeout(travel)
                            self.stats['truck_op'] += travel
                            
                            # Wait Loop at Go Line
                            while not self.shovel_agent.is_available:
                                yield self.env.timeout(5)
                                self.stats['truck_standby'] += 5
                                
                            # Drive Back
                            travel_back = max(0, random.gauss(GO_LINE_EMPTY, GO_LINE_EMPTY_STD))
                            yield self.env.timeout(travel_back)
                            self.stats['truck_op'] += travel_back
                            
                            # Restart Cycle (Try to join queue again)
                            break 
                        else:
                            # MTTR is short. Stay in queue.
                            # We loop back to `yield req | breakdown_event`
                            # But we must ensure we don't catch the SAME event again
                            # (SimPy events are single-shot, so shovel makes a new one)
                            pass

            except simpy.Interrupt:
                # Failsafe
                req.cancel()
                joined_queue = False

            if not joined_queue:
                continue

            #we will wait for either the request to be granted or the 
            # 6. Queue
            current_time = self.env.now
            total_wait_time = current_time - arrive_time
            delay_deduction = 0.0
            if self.shovel_agent.last_break_end_time > arrive_time:
                overlap = self.shovel_agent.last_break_end_time - arrive_time
                delay_deduction = min(total_wait_time, overlap)
            
            if delay_deduction > 0:
                self.stats['truck_standby'] += delay_deduction
            self.stats['truck_queue'] += (total_wait_time - delay_deduction)

            #7 Manuver
            maneuver_time = D_MANUVER.get_duration()
            yield self.env.timeout(maneuver_time)
            self.stats['truck_op'] += maneuver_time
            self.shovel_agent.log_op(maneuver_time)            


            #8 Loading
            load_time = D_LOAD.get_duration()
            yield self.env.timeout(load_time)
            self.stats['truck_op'] += load_time
            self.shovel_agent.log_op(load_time)            

            # 9 Depart Shovel
            self.shovel_agent.truck_departed()
            self.shovel_agent.res.release(req) # Explicit release

            #10 Log Tonnes
            payload = random.gauss(TRUCK_PAYLOAD_MEAN, TRUCK_PAYLOAD_STD)
            self.stats['tonnes'] += payload
            self.stats['loads'] += 1
            self.shovel_agent.log_tonnes(payload)            

            # 7. Haul
            haul_time = max(0, random.gauss(self.shovel_agent.config.haul_mean, self.shovel_agent.config.haul_std))
            yield self.env.timeout(haul_time)
            self.stats['truck_op'] += haul_time
            
            # 8. Dump
            dump_time = D_DUMP.get_duration()
            yield self.env.timeout(dump_time)
            self.stats['truck_op'] += dump_time

            # 9. Fuel
            if self.needs_fuel and not self.has_fueled and self.env.now >= self.fuel_trigger_time:
                fuel_time = D_FUEL.get_duration()
                self.stats['truck_standby'] += fuel_time
                yield self.env.timeout(fuel_time) 
                self.has_fueled = True
            
            # 10. Return - with a check to make sure their shovel is still up
            if not self.shovel_agent.is_available:
                travel_to_goline = max(0, random.gauss(GO_LINE_EMPTY,GO_LINE_EMPTY_STD))
                yield self.env.timeout(travel_to_goline)
                self.stats['truck_op'] += travel_to_goline
                while not self.shovel_agent.is_available:
                    yield self.env.timeout(5)
                    self.stats['truck_standby'] += 5 
                travel_from_goline = max(0,random.gauss(GO_LINE_EMPTY, GO_LINE_EMPTY_STD))
                yield self.env.timeout(travel_from_goline)
                self.stats['truck_op'] += travel_from_goline
            else: # return normally
                return_time = max(0, random.gauss(self.shovel_agent.config.return_mean, self.shovel_agent.config.return_std))
                yield self.env.timeout(return_time)
                self.stats['truck_op'] += return_time

#Reporting
def print_shift_header(data):
    # Header spanning 14 columns
    header = (
        f"{'Day':<4} {'Shift':<6} "
        f"{'Avail':<6} {'Tonnes':<8} "
        f"{'TrkAv%':<7} {'TrkUt%':<7} {'TrkEf%':<7} {'Q/Trk':<7} "
        f"{'ShvAv':<6} {'ShvAv%':<7} {'ShvUt%':<7} {'ShvEf%':<7} {'Hang/Shv':<9}"
    )
    print("-" * len(header))
    print(header)
    print("-" * len(header))

def print_shift_row(s, data):
    t_cal = NUM_TRUCKS_TOTAL * SIM_MINUTES
    
    # Available Time = Calendar - Unplanned Down - Planned Down
    t_down_total = data['truck_down'] + data['planned_down']
    t_avail_time = t_cal - t_down_total
    
    avg_trucks_avail = t_avail_time / SIM_MINUTES
    
    # Engine On (SMU) = Operating + Queue + Standby
    t_smu = data['truck_op'] + data['truck_queue'] + data['truck_standby']
    
    # Operating (Productive) = Active + Queue
    # (Queueing is required to get a load, therefore it is productive time)
    t_operating = data['truck_op'] + data['truck_queue']
    
    trk_avail_pct = (t_avail_time / t_cal) * 100
    trk_util_pct  = (t_smu / t_avail_time) * 100 if t_avail_time > 0 else 0
    
    # FIX: Efficiency = Productive (Op + Queue) / Total Engine On (SMU)
    trk_eff_pct   = (t_operating / t_smu) * 100 if t_smu > 0 else 0
    
    q_per_truck   = data['truck_queue'] / avg_trucks_avail if avg_trucks_avail > 0 else 0

    # Shovels
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

    day = (s // 2) + 1
    shft = "D" if s % 2 == 0 else "N"
    
    print(
        f"{day:<4} {shft:<6} "
        f"{avg_trucks_avail:<6.1f} {data['tonnes']:<8,.0f} "
        f"{trk_avail_pct:<7.1f} {trk_util_pct:<7.1f} {trk_eff_pct:<7.1f} {q_per_truck:<7.1f} "
        f"{avg_shovels_avail:<6.1f} {shv_avail_pct:<7.1f} {shv_util_pct:<7.1f} {shv_eff_pct:<7.1f} {hang_per_shovel:<9.0f}"
    )

def print_final_report(shift_data_list, df_pm, planned_down_hours):
    #aggregation
    agg = {k: sum(d[k] for d in shift_data_list) for k in shift_data_list[0] if k in ['tonnes', 'loads', 'truck_op', 'truck_queue', 'truck_standby', 'truck_parked', 'truck_down', 'shovel_op', 'shovel_cal', 'shovel_down', 'shovel_parked']}  
    hrs_active  = agg['truck_op']/60
    hrs_queue   = agg['truck_queue']/60
    hrs_standby = agg['truck_standby']/60
    hrs_parked  = agg['truck_parked']/60
    hrs_operating = hrs_active + hrs_queue
    hrs_smu = hrs_operating + hrs_standby

    hrs_cal = (NUM_TRUCKS_TOTAL * TOTAL_SHIFTS * SHIFT_HOURS)
    hrs_down= (agg['truck_down']/60) + planned_down_hours
    hrs_avail = hrs_cal - hrs_down

    shovel_smu_hrs = (agg['shovel_op'] / 60) + max(0, (agg['shovel_cal']/60) - (agg['shovel_down']/60) - (agg['shovel_parked']/60) - (agg['shovel_op']/60))
    monthly_mult = 30.4 / DAYS_TO_RUN
    annual_mult  = 365 / DAYS_TO_RUN

    print("\n" + "="*80)
    print(" PERFORMANCE SUMMARY (FLEET)")
    print("="*80)    

    summary_data = [
        {"Metric": "Total Tonnes",     "Unit": "t",       "Sim": agg['tonnes'],  "Monthly": agg['tonnes']*monthly_mult, "Annual": agg['tonnes']*annual_mult},
        {"Metric": "Operating Hrs",    "Unit": "hrs",     "Sim": hrs_operating,  "Monthly": hrs_operating*monthly_mult, "Annual": hrs_operating*annual_mult},
        {"Metric": "  - Active",       "Unit": "hrs",     "Sim": hrs_active,     "Monthly": "-", "Annual": "-"},
        {"Metric": "  - Queue",        "Unit": "hrs",     "Sim": hrs_queue,      "Monthly": "-", "Annual": "-"},
        {"Metric": "Standby Hrs",      "Unit": "hrs",     "Sim": hrs_standby,    "Monthly": "-", "Annual": "-"},
        {"Metric": "Parked Hrs",       "Unit": "hrs",     "Sim": hrs_parked,     "Monthly": "-", "Annual": "-"},
        {"Metric": "Truck Avail",      "Unit": "%",       "Sim": (hrs_avail/hrs_cal)*100, "Monthly": "-", "Annual": "-"},
        {"Metric": "Asset Util",       "Unit": "%",       "Sim": (hrs_smu/hrs_avail)*100, "Monthly": "-", "Annual": "-"},
        {"Metric": "Effective Util",   "Unit": "%",       "Sim": (hrs_active/hrs_operating)*100, "Monthly": "-", "Annual": "-"},
        {"Metric": "Fleet Prod",       "Unit": "t/OpHr",  "Sim": agg['tonnes']/hrs_operating, "Monthly": "-", "Annual": "-"},
    ]    

    df_sum = pd.DataFrame(summary_data)
    pd.options.display.float_format = '{:,.1f}'.format
    print(df_sum.to_string(index=False))

# --- SHOVEL REPORT (PER SHOVEL) ---
    print("\n" + "="*80)
    print(" SHOVEL PERFORMANCE REPORT")
    print("="*80)
    
    # Aggregate per-shovel stats from all shifts
    shovel_totals = {}
    
    for d in shift_data_list:
        for name, stats in d['shovel_perf'].items():
            if name not in shovel_totals:
                shovel_totals[name] = {'tonnes': 0, 'down': 0, 'planned_down': 0, 'op': 0, 'hang': 0, 'cal': 0}
            
            shovel_totals[name]['tonnes'] += stats['tonnes']
            shovel_totals[name]['down']   += stats['down']
            shovel_totals[name]['planned_down'] += stats['planned_down']
            shovel_totals[name]['op']     += stats['op']
            shovel_totals[name]['hang']   += stats['hang']
            shovel_totals[name]['cal']    += SIM_MINUTES

    shv_report_data = []
    
    for name, s in shovel_totals.items():
        # Calculations (in Hours)
        cal_hrs = s['cal'] / 60
        down_hrs = (s['down'] + s['planned_down']) / 60
        op_hrs = s['op'] / 60
        hang_hrs = s['hang'] / 60
        
        # Availability
        avail_pct = ((cal_hrs - down_hrs) / cal_hrs) if cal_hrs > 0 else 0
        
        # Utilization (of Available Time)
        # Note: Op + Hang = Utilized Time
        avail_hrs = cal_hrs - down_hrs
        util_pct = ((op_hrs + hang_hrs) / avail_hrs) if avail_hrs > 0 else 0
        
        # Operating Hours (User Formula: Cal * Avail * Util)
        # Mathematical Equivalent: Op + Hang
        user_op_hours = op_hrs + hang_hrs
        
        # Productivity
        prod = s['tonnes'] / user_op_hours if user_op_hours > 0 else 0
        
        shv_report_data.append({
            "Shovel ID": name,
            "Tonnes": s['tonnes'],
            "Avail %": avail_pct * 100,
            "Util %": util_pct * 100,
            "Op Hrs": user_op_hours,
            "Prod (t/hr)": prod,
            "Hang Hrs": hang_hrs
        })

    df_shv = pd.DataFrame(shv_report_data)
    print(df_shv.to_string(index=False))

# --- D. COST TABLE ---
    print("\n" + "-"*80)
    print(" COST PROJECTION")
    print("-"*80)
    
    cost_data = []
    for name, smu, rate in [("Shovel Fleet", shovel_smu_hrs, COST_SHOVEL_SMU), 
                            ("Truck Fleet",  hrs_smu,        COST_TRUCK_SMU)]:
        total_sim = smu * rate
        cost_per_t = total_sim / agg['tonnes']
        cost_data.append({
            "Fleet": name,
            "$/SMU Rate": rate,
            "$/SMU (Sim)": rate, 
            "Total $ (Sim)": total_sim,
            "Total $ (Mth)": total_sim * monthly_mult,
            "$/t (Unit Cost)": f"{cost_per_t:.2f}"
        })
        
    df_cost = pd.DataFrame(cost_data)
    print(df_cost.to_string(index=False))
    print("\n" + "="*80 + "\n")

# --- 4. Core DISPATCH LOGIC --- "the Planner"

def run_shift(shift_num, pm_schedule):
    env = simpy.Environment()
    
    # dictionary
    stats = {
        'tonnes': 0, 'loads': 0,
        'truck_op': 0.0, 'truck_queue': 0.0, 'truck_standby': 0.0, 'truck_parked': 0.0, 
        'truck_down': 0.0, 'planned_down': 0.0,
        'shovel_op': 0.0, 'shovel_down': 0.0, 'shovel_planned_down': 0.0, 
        'shovel_parked': 0.0, 'shovel_cal': 0.0, 
        'dispatch_log': [],
        'hang_breakdown': {},
        'shovel_perf': {}, 
        'planned_down_ids': [] 
    }    
    
    is_blast = (random.random() < BLAST_CHANCE)

    # --- 1. MAINTENANCE LINEUP ---
    shift_start_hr = shift_num * SHIFT_HOURS
    shift_end_hr = (shift_num + 1) * SHIFT_HOURS

    # TRUCK PM FILTER
    trucks_down_for_pm = set()
    if not pm_schedule.empty:
        # Filter for TRUCKS only
        relevant_truck_pms = pm_schedule[
            (pm_schedule["Asset Type"] == "TRUCK") &
            (pm_schedule["Hours_Into_Sim"] < shift_end_hr) & 
            ((pm_schedule["Hours_Into_Sim"] + pm_schedule["Downtime"]) > shift_start_hr)
        ]
        for _, row in relevant_truck_pms.iterrows():
            trucks_down_for_pm.add(row["Asset ID"])
            stats['planned_down_ids'].append(f"{row['Asset ID']} ({row['Event']})") 

    # SHOVEL PM FILTER
    shovel_pm_map = {} 
    if not pm_schedule.empty:
        relevant_shovel_pms = pm_schedule[
            (pm_schedule["Asset Type"] == "SHOVEL") &
            (pm_schedule["Hours_Into_Sim"] < shift_end_hr) & 
            ((pm_schedule["Hours_Into_Sim"] + pm_schedule["Downtime"]) > shift_start_hr)
        ]
        for _, row in relevant_shovel_pms.iterrows():
            start_relative = max(0, row["Hours_Into_Sim"] - shift_start_hr) * 60
            duration = row["Downtime"] * 60
            shovel_pm_map[row["Asset ID"]] = (start_relative, duration)
            stats['dispatch_log'].append(f"PLANNED DOWN | {row['Asset ID']} ({row['Event']}) - Duration: {row['Downtime']}h")

    available_truck_pool = []
    for i in range(NUM_TRUCKS_TOTAL):
        t_id = f"T-{i}" 
        
        if t_id not in trucks_down_for_pm:
            available_truck_pool.append(t_id)
        else:
            # If the truck is held back, we lose 12 hours of availability
            stats['planned_down'] += SIM_MINUTES 

    remaining_trucks = len(available_truck_pool)
    
    stats['dispatch_log'].append(f"FLEET STATUS | Total: {NUM_TRUCKS_TOTAL} | Down: {len(trucks_down_for_pm)} | Avail: {remaining_trucks}")
    if len(trucks_down_for_pm) > 0:
        stats['dispatch_log'].append(f"PLANNED DOWN | {', '.join(sorted(list(trucks_down_for_pm)))}")

    # --- 2. DISPATCH ASSIGNMENT ---
    active_configs = list(SHOVEL_FLEET) 
    active_configs.sort(key=lambda x: x.priority)
    shovel_lunch_times = generate_lunch_schedule(active_configs, LUNCH_WIN_START, LUNCH_WIN_END, D_LUNCH)
    created_agents = {}

    EST_LOAD_TIME = D_LOAD.mean
    EST_DUMP_TIME = D_DUMP.mean
    EST_MAN = D_MANUVER.mean

    for cfg in active_configs:
        cycle_time = (EST_MAN + EST_LOAD_TIME + cfg.haul_mean + EST_DUMP_TIME + cfg.return_mean + EST_QUEUE_BUFFER)
        service_time = EST_LOAD_TIME + EST_MAN
        trucks_needed = math.ceil(cycle_time / service_time)
        
        assigned_count = min(remaining_trucks, trucks_needed)
        #is required?
        is_req = (assigned_count>0)
        my_lunch_time = shovel_lunch_times[cfg]
        pm_data = shovel_pm_map.get(cfg.name, None)
        agent = ShovelAgent(env, cfg, stats, is_blast, my_lunch_time, pm_data, is_required=is_req)
        created_agents[cfg.name] = agent 
        stats['shovel_cal'] += SIM_MINUTES 

        if is_req: #only assign trucks if required
            for _ in range(assigned_count):
                truck_id = available_truck_pool.pop(0) 
                TruckAgent(env, truck_id, agent, stats)
            
            remaining_trucks -= assigned_count
            stats['dispatch_log'].append(f"{cfg.name:<15} | Cycle: {cycle_time:.1f}m | Req: {trucks_needed} | Assigned: {assigned_count}")
        else:
             stats['dispatch_log'].append(f"{cfg.name:<15} | Cycle: {cycle_time:.1f}m | Req: {trucks_needed} | Assigned: 0 (PARKED)")

    # --- 3. SPARES ---
    if len(available_truck_pool) > 0:
        stats['truck_parked'] += (len(available_truck_pool) * SIM_MINUTES)
        stats['dispatch_log'].append(f"{'STANDBY FLEET':<15} | Count: {len(available_truck_pool)} Trucks (Engine Off / Spare)")
    
    if is_blast:
        stats['dispatch_log'].append(f"*** BLAST SCHEDULED THIS SHIFT ***")

    # --- 4. RUN SIMULATION ---
    env.run(until=SIM_MINUTES)

    # --- DEBUG GRAPH GENERATION ---
    if DEBUG_MODE:
        # We pass the dictionary of shovel agents we created this shift
        generate_shovel_debug_graph(created_agents, shift_num)    
    
# --- POST SHIFT SHOVEL EXTRACTION ---
    for cfg in SHOVEL_FLEET:
        if cfg.name in created_agents:
            agent = created_agents[cfg.name]
            # Hang Time Logic
            hang = agent.my_cal - agent.my_down - agent.my_planned_down - agent.my_parked - agent.my_op
            hang = max(0, hang)
            stats['hang_breakdown'][cfg.name] = hang
            
            # Detailed Breakdown for Final Report
            stats['shovel_perf'][cfg.name] = {
                'tonnes': agent.my_tonnes,
                'op': agent.my_op,
                'down': agent.my_down,
                'planned_down': agent.my_planned_down,
                'hang': hang,
                'parked': agent.my_parked
            }
        else:
            # Shovel existed but wasn't assigned/created (Parked or Down?)
            stats['hang_breakdown'][cfg.name] = 0.0
            # If shovel had a PM preventing it from starting, handle logic here?
            # For now assume parked/unused.
            stats['shovel_perf'][cfg.name] = {'tonnes': 0, 'op': 0, 'down': 0, 'planned_down': 0, 'hang': 0, 'parked': SIM_MINUTES}
    return stats


# --- 6. PM SCHEDULING LOGIC (Simulation Scope Only) ---
def generate_fleet_schedule(num_trucks, sim_duration_hours):
    all_events = []
    print(f"\n=== GENERATING PM PLAN (Scope: {sim_duration_hours} Hours) ===")

    # --- 1. TRUCK PMs ---
    for i in range(num_trucks): 
        truck_id = f"T-{i}"     
        start_hours = random.randint(MIN_SMU_HRS, MAX_SMU_HRS)
        end_hours = start_hours + sim_duration_hours
        current_pm_target = ((start_hours // 250) + 1) * 250
        
        while current_pm_target <= end_hours:
            if current_pm_target % 1000 == 0:
                pm_type = "C-Check (12h)"; duration = 12
            elif current_pm_target % 500 == 0:
                pm_type = "B-Check (8h)"; duration = 8
            else:
                pm_type = "A-Check (4h)"; duration = 4
            
            all_events.append({
                "Asset ID": truck_id, "Asset Type": "TRUCK",
                "Start Hours": start_hours, "PM Interval": current_pm_target, 
                "Event": pm_type, "Downtime": duration, "Labor Cost": duration * COST_PER_DOWNTIME_HR
            })
            current_pm_target += 250

    # --- 2. SHOVEL PMs ---
    for shovel in SHOVEL_FLEET:
        # Shovels are usually younger/varied, lets say 20k-40k hours
        start_hours = random.randint(SHVL_MIN_SMU_HRS,SHVL_MAX_SMU_HRS)
        end_hours = start_hours + sim_duration_hours
        current_pm_target = ((start_hours // 250) + 1) * 250
        
        while current_pm_target <= end_hours:
            if current_pm_target % 1000 == 0:
                pm_type = "C-Check (12h)"; duration = 12
            elif current_pm_target % 500 == 0:
                pm_type = "B-Check (8h)"; duration = 8
            else:
                pm_type = "A-Check (4h)"; duration = 4
            
            all_events.append({
                "Asset ID": shovel.name, "Asset Type": "SHOVEL",
                "Start Hours": start_hours, "PM Interval": current_pm_target, 
                "Event": pm_type, "Downtime": duration, "Labor Cost": duration * COST_PER_DOWNTIME_HR
            })
            current_pm_target += 250

    if not all_events:
        print("No PM events scheduled.")
        return pd.DataFrame(columns=["Asset ID", "Asset Type", "Start Hours", "PM Interval", "Event", "Downtime", "Labor Cost"])

    return pd.DataFrame(all_events)

#debugging Graph 1 - shovel status
def generate_shovel_debug_graph(shovels_dict, shift_num):
#    print(f"   -> Saving debug graph for Shift {shift_num}...")
    status_map = {'shovel_parked': 1, 'shovel_down': 2, 'shovel_op': 3}

    num_shovels = len(shovels_dict)
    fig, axs = plt.subplots(num_shovels, 1, figsize=(12, 4*num_shovels), sharex=True)
    if num_shovels == 1: axs = [axs]

    for i, (shovel_name, shovel_agent) in enumerate(shovels_dict.items()):
        # 1. Plot Status (Left Axis)
        times_s = []
        statuses = []
        history_s = shovel_agent.state_history[:]
        history_s.append((SIM_MINUTES, history_s[-1][1]))
        for entry in history_s:
            times_s.append(entry[0])
            statuses.append(status_map.get(entry[1], 3))

        ax1 = axs[i]
        ax1.step(times_s, statuses, where='post', color='red', linewidth=2, label='Status')
        ax1.set_ylabel("Shovel Status", color='red', fontweight='bold')
        ax1.set_ylim(0.5, 3.5)
        ax1.set_yticks([1, 2, 3])
        ax1.set_yticklabels(['Parked', 'Down', 'Op'])
        ax1.tick_params(axis='y', labelcolor='red')
        ax1.grid(True, axis='x', linestyle='--', alpha=0.7)
        ax1.set_title(f"Shovel Status & Queue: {shovel_name}", fontsize=12, fontweight='bold')

        # Color Zones
        ax1.axhspan(2.5, 3.5, facecolor='green', alpha=0.1)
        ax1.axhspan(1.5, 2.5, facecolor='orange', alpha=0.1)
        ax1.axhspan(0.5, 1.5, facecolor='gray', alpha=0.1)

        # 2. Plot Queue Count (Right Axis)
        ax2 = ax1.twinx()
        times_q = []
        counts = []
        history_q = shovel_agent.queue_history[:]
        history_q.append((SIM_MINUTES, history_q[-1][1]))
        
        for entry in history_q:
            times_q.append(entry[0])
            counts.append(entry[1])

        ax2.step(times_q, counts, where='post', color='blue', linewidth=1.5, linestyle='-', alpha=0.7, label='Queue')
        ax2.set_ylabel("Trucks @ Shovel", color='blue', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='blue')
        
        # Dynamic Y-limit for Queue to keep it readable
        max_q = max(counts) if counts else 0
        ax2.set_ylim(0, max(5, max_q + 2)) 

    plt.xlabel("Shift Time (minutes)")
    plt.xlim(0, SIM_MINUTES)
    plt.tight_layout()
    filename = f"shift_{shift_num:03d}_debug.png"
    plt.savefig(filename)
    plt.close(fig)




# --- Running the simulation
def main():
    sim_duration = TOTAL_SHIFTS * SHIFT_HOURS
    print(f"=== Simulation Setup: {TOTAL_SHIFTS} Shifts ({sim_duration} Hours) ===")

    # 1. GENERATE THE PLAN (Simulation Scope Only)
    # ---------------------------------------------------------
    df_pm = generate_fleet_schedule(NUM_TRUCKS_TOTAL, sim_duration)
    
    # Calculate "Sim Hours" (Relative time from start of sim)
    if not df_pm.empty:
        df_pm["Hours_Into_Sim"] = df_pm["PM Interval"] - df_pm["Start Hours"]
        df_pm["Planned Shift"] = (df_pm["Hours_Into_Sim"] // SHIFT_HOURS).astype(int) + 1
        planned_down_hours = df_pm["Downtime"].sum()
    else:
        df_pm["Hours_Into_Sim"] = 0
        planned_down_hours = 0.0

    print(f"=== MAINTENANCE PLAN GENERATED ===")
    print(f"Total Events: {len(df_pm)} | Total Downtime: {planned_down_hours:.1f} Hours")
    
    if len(df_pm) > 0:
        print("-" * 75)
        print(f"{'Asset ID':<10} {'Sim Hour':<10} {'Est. Shift':<12} {'Event Type':<15} {'Duration':<10}")
        print("-" * 75)
        for _, row in df_pm.sort_values("Hours_Into_Sim").iterrows():
            print(f"{row['Asset ID']:<10} {row['Hours_Into_Sim']:<10.1f} {row['Planned Shift']:<12} {row['Event']:<15} {row['Downtime']:<10.1f}")
        print("-" * 75)   

    print(f"\n=== STARTING SIMULATION FOR {TOTAL_SHIFTS} SHIFTS ===")

    # 2. RUN SIMULATION LOOP
    # ---------------------------------------------------------
    shift_data_list = []
    
    for s in range(TOTAL_SHIFTS):
        data = run_shift(s, df_pm) 
        shift_data_list.append(data)
        
        if s == 0:
            print_shift_header(data)
        print_shift_row(s, data)

    # 3. GENERATE FINAL REPORT
    # ---------------------------------------------------------
    print_final_report(shift_data_list, df_pm, planned_down_hours)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "go":
        if len(sys.argv) > 2 and sys.argv[2] =="1":
            DEBUG_MODE = True
            print("-----Debug is on-----")
        main()
    else:
        print("\n" + "="*80)
        print(f"{'MINING DISPATCH SIMULATOR - LOGIC & USAGE GUIDE':^80}")
        print("="*80)
        
        print("\n--- 1. OVERVIEW ---")
        print("This is an Event-Driven Simulation (using SimPy) that models a Load & Haul")
        print("mining circuit. Unlike a spreadsheet, it simulates interactions (queuing,")
        print("bunching, hang time) and random variation to predict realistic performance.")

        print("\n--- 2. REPORT LEGEND & DEFINITIONS ---")
        print("  Naming Convention:")
        print("    T-xx  = Truck (e.g., T-01)")
        print("    EXxx  = Excavator/Shovel (e.g., EX01)")
        print("\n  Output Columns:")
        print(f"  {'Column':<10} {'Unit':<10} {'Definition':<50}")
        print("  " + "-"*70)
        print(f"  {'Avail':<10} {'Count':<10} {'Avg number of Trucks physically available (not Down)'}")
        print(f"  {'Tonnes':<10} {'Tonnes':<10} {'Total production moved this shift'}")
        print(f"  {'TrkAv%':<10} {'%':<10} {'Availability: (Calendar Time - Down) / Calendar'}")
        print(f"  {'TrkUt%':<10} {'%':<10} {'Utilization: (Operating + Standby) / Available'}")
        print(f"  {'TrkEf%':<10} {'%':<10} {'Efficiency: (Operating Time) / (Operating + Standby)'}")
        print(f"  {'':<10} {'':<10} {'*Note: Operating = Active + Queue (Productive)'}")
        print(f"  {'Q/Trk':<10} {'Minutes':<10} {'Queue Time: Avg minutes a truck spent waiting in line'}")
        print(f"  {'ShvAv':<10} {'Count':<10} {'Avg number of Shovels available'}")
        print(f"  {'ShvUt%':<10} {'%':<10} {'Utilization: (Operating + Hang) / Available'}")
        print(f"  {'Hang/Shv':<10} {'Minutes':<10} {'Hang Time: Avg minutes a shovel sat waiting for trucks'}")

        print("\n--- 3. TIME CLASSIFICATIONS (Definitions) ---")
        print(f"{'Category':<15} | {'Description':<50}")
        print("-" * 70)
        print(f"{'OPERATING':<15} | Engine ON and working. Includes:")
        print(f"{'':<15} | - Active: Hauling, Loading, Dumping, Maneuvering")
        print(f"{'':<15} | - Queue:  Waiting at Shovel (Engine Running)")
        print("-" * 70)
        print(f"{'STANDBY':<15} | Engine ON but not productive. Includes:")
        print(f"{'':<15} | - Smoko Break (Short)")
        print(f"{'':<15} | - Refueling")
        print(f"{'':<15} | - Start of Shift (Pre-start/Tramming)")
        print("-" * 70)
        print(f"{'PARKED':<15} | Engine OFF. Includes:")
        print(f"{'':<15} | - Lunch Break (Long)")
        print(f"{'':<15} | - Blast Delays")
        print(f"{'':<15} | - End of Shift / Spare Truck (Not Required)")
        print("-" * 70)
        print(f"{'DOWN':<15} | Mechanical Failure. Includes:")
        print(f"{'':<15} | - Unplanned Breakdowns (Random)")
        print(f"{'':<15} | - Planned Maintenance (A/B/C Checks)")

        print("\n--- 4. STATISTICAL MODELS (How Randomness Works) ---")
        print("Real mining isn't static. This simulator uses three types of randomness:")
        print("  A. GAUSSIAN (Normal Distribution): Used for Human/Machine Variance.")
        print("     - Example: Load Time, Haul Time, Payload.")
        print("     - Why: Most cycles are near the average, with some faster/slower.")
        
        print("  B. EXPONENTIAL (Poisson Process): Used for Reliability (MTBF).")
        print("     - Example: Time Between Failures.")
        print("     - Why: Failures are random events. A hose can burst at hour 1 or hour 500.")
        
        print("  C. UNIFORM (Flat Distribution): Used for Bounded Uncertainty.")
        print("     - Example: Break Times, Failure Timing, Fuel Trigger.")
        print("     - Why: A breakdown is equally likely to happen at 8am vs 2pm.")

        print("\n--- 5. GENERAL CONFIGURATION ---")
        print("  To Change Settings: Open the script and edit Section 1 (Configuration).")
        print("     - NUM_TRUCKS_TOTAL: Change fleet size.")
        print("     - SHOVEL_FLEET:     Add/Remove shovels or change cycle times.")
        print("     - DAYS_TO_RUN:      Increase for better long-term averages.")

        print("\n--- 6. MAINTENANCE CONFIGURATION (Important) ---")
        print("  A. UNPLANNED BREAKDOWNS (Random Events)")
        print("     To change how often things break randomly, edit these variables:")
        print("       - PROF_TRUCK_MTBF  (Mean Time Between Failure)")
        print("       - PROF_TRUCK_MTTR  (Mean Time To Repair)")
        
        print("\n  B. PLANNED MAINTENANCE (Scheduled A/B/C Services)")
        print("     To change Service Intervals (250/500/1000) or Durations (4h/8h/12h),")
        print("     you must edit the 'generate_fleet_schedule' function directly.")
        print("     Look for the logic: 'if current_pm_target % 1000 == 0: duration = 12'")

        print("\n--- 7. ENVIRONMENT SETUP (For New Users) ---")
        print("  If you get an error saying 'No module named simpy', follow these steps:")
        print("  1. Install Python (3.8+ recommended).")
        print("  2. Create a virtual environment:")
        print("       Linux/Mac:   python3 -m venv simenv")
        print("       Windows:     python -m venv simenv")
        print("  3. Activate the environment:")
        print("       Linux/Mac:   source simenv/bin/activate")
        print("       Windows:     simenv\\Scripts\\activate")
        print("  4. Install libraries: pip install simpy pandas")
        
        print("\n  To Run the Simulation:")
        print("     python LHSimulatorDispatch.py go")
        print("="*80 + "\n")
