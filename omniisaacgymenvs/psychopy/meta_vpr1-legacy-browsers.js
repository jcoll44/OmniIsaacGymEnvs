/****************** 
 * Meta_Vpr1 Test *
 ******************/


// store info about the experiment session:
let expName = 'meta_vpr1';  // from the Builder filename that created this script
let expInfo = {
    'participant': 'make up a name',
    'session': '001',
};

// Start code blocks for 'Before Experiment'
// init psychoJS:
const psychoJS = new PsychoJS({
  debug: true
});

// open window:
psychoJS.openWindow({
  fullscr: true,
  color: new util.Color([0,0,0]),
  units: 'height',
  waitBlanking: true
});
// schedule the experiment:
psychoJS.schedule(psychoJS.gui.DlgFromDict({
  dictionary: expInfo,
  title: expName
}));

const flowScheduler = new Scheduler(psychoJS);
const dialogCancelScheduler = new Scheduler(psychoJS);
psychoJS.scheduleCondition(function() { return (psychoJS.gui.dialogComponent.button === 'OK'); }, flowScheduler, dialogCancelScheduler);

// flowScheduler gets run if the participants presses OK
flowScheduler.add(updateInfo); // add timeStamp
flowScheduler.add(experimentInit);
flowScheduler.add(instructionsRoutineBegin());
flowScheduler.add(instructionsRoutineEachFrame());
flowScheduler.add(instructionsRoutineEnd());
const practice_trialsLoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(practice_trialsLoopBegin(practice_trialsLoopScheduler));
flowScheduler.add(practice_trialsLoopScheduler);
flowScheduler.add(practice_trialsLoopEnd);
flowScheduler.add(begin_exptRoutineBegin());
flowScheduler.add(begin_exptRoutineEachFrame());
flowScheduler.add(begin_exptRoutineEnd());
const trialsLoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(trialsLoopBegin(trialsLoopScheduler));
flowScheduler.add(trialsLoopScheduler);
flowScheduler.add(trialsLoopEnd);
flowScheduler.add(thanksRoutineBegin());
flowScheduler.add(thanksRoutineEachFrame());
flowScheduler.add(thanksRoutineEnd());
flowScheduler.add(quitPsychoJS, '', true);

// quit if user presses Cancel in dialog box:
dialogCancelScheduler.add(quitPsychoJS, '', false);

psychoJS.start({
  expName: expName,
  expInfo: expInfo,
  resources: [
    {'name': '../recordings/17/first_frame.png', 'path': '../recordings/17/first_frame.png'},
    {'name': '../recordings/16/first_frame.png', 'path': '../recordings/16/first_frame.png'},
    {'name': '../recordings/14/first_frame.png', 'path': '../recordings/14/first_frame.png'},
    {'name': '../recordings/6/first_frame.png', 'path': '../recordings/6/first_frame.png'},
    {'name': '../recordings/11/first_frame.png', 'path': '../recordings/11/first_frame.png'},
    {'name': 'vpr_stimuli1.xlsx', 'path': 'vpr_stimuli1.xlsx'},
    {'name': '../recordings/13/first_frame.png', 'path': '../recordings/13/first_frame.png'},
    {'name': '../recordings/1/vid.mp4', 'path': '../recordings/1/vid.mp4'},
    {'name': '../recordings/7/first_frame.png', 'path': '../recordings/7/first_frame.png'},
    {'name': '../recordings/18/first_frame.png', 'path': '../recordings/18/first_frame.png'},
    {'name': '../recordings/12/first_frame.png', 'path': '../recordings/12/first_frame.png'},
    {'name': '../recordings/15/first_frame.png', 'path': '../recordings/15/first_frame.png'},
    {'name': '../recordings/19/first_frame.png', 'path': '../recordings/19/first_frame.png'},
    {'name': '../recordings/5/first_frame.png', 'path': '../recordings/5/first_frame.png'},
    {'name': '../recordings/2/first_frame.png', 'path': '../recordings/2/first_frame.png'},
    {'name': '../recordings/3/first_frame.png', 'path': '../recordings/3/first_frame.png'},
    {'name': '../recordings/10/first_frame.png', 'path': '../recordings/10/first_frame.png'},
    {'name': '../recordings/1/first_frame.png', 'path': '../recordings/1/first_frame.png'},
    {'name': '../recordings/4/first_frame.png', 'path': '../recordings/4/first_frame.png'},
    {'name': '../recordings/8/first_frame.png', 'path': '../recordings/8/first_frame.png'},
    {'name': '../recordings/9/first_frame.png', 'path': '../recordings/9/first_frame.png'}
  ]
});

psychoJS.experimentLogger.setLevel(core.Logger.ServerLevel.EXP);


var currentLoop;
var frameDur;
async function updateInfo() {
  currentLoop = psychoJS.experiment;  // right now there are no loops
  expInfo['date'] = util.MonotonicClock.getDateStr();  // add a simple timestamp
  expInfo['expName'] = expName;
  expInfo['psychopyVersion'] = '2022.2.5';
  expInfo['OS'] = window.navigator.platform;


  // store frame rate of monitor if we can measure it successfully
  expInfo['frameRate'] = psychoJS.window.getActualFrameRate();
  if (typeof expInfo['frameRate'] !== 'undefined')
    frameDur = 1.0 / Math.round(expInfo['frameRate']);
  else
    frameDur = 1.0 / 60.0; // couldn't get a reliable measure so guess

  // add info from the URL:
  util.addInfoFromUrl(expInfo);
  

  
  psychoJS.experiment.dataFileName = (("." + "/") + `data/${expInfo["participant"]}_${expName}_${expInfo["date"]}`);


  return Scheduler.Event.NEXT;
}


var instructionsClock;
var text_inst;
var key_resp_inst;
var fixateClock;
var fixate_middle;
var text;
var videoClock;
var BinaryClock;
var Goal_reached;
var key_resp;
var begin_exptClock;
var text_mid;
var key_resp_mid;
var chooseClock;
var image;
var Binary2Clock;
var Question;
var yes_no;
var confidenceClock;
var text_confidence;
var key_resp_confidence;
var thanksClock;
var text_thanks;
var key_resp_thanks;
var globalClock;
var routineTimer;
async function experimentInit() {
  // Initialize components for Routine "instructions"
  instructionsClock = new util.Clock();
  text_inst = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_inst',
    text: "In this experiment, you will be shown several videos and images and asked a series of questions. The first task is to watch videos of a robot attempt to reach an orange cone and then answer after each video a question about the robot's performance.\n\nThe second task is to look at an image of the robot start state and decide if the robot will reach the cone. You will be asked to answer whether the robot would reach the cone, yes/no, and what is you confidence in your response. \n\nPress any key to begin.",
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0], height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  key_resp_inst = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "fixate"
  fixateClock = new util.Clock();
  fixate_middle = new visual.TextStim({
    win: psychoJS.window,
    name: 'fixate_middle',
    text: '+',
    font: 'Open Sans',
    units: undefined, 
    pos: [0.0, 0], height: 0.1,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  text = new visual.TextStim({
    win: psychoJS.window,
    name: 'text',
    text: '',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0], height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -1.0 
  });
  
  // Initialize components for Routine "video"
  videoClock = new util.Clock();
  // Initialize components for Routine "Binary"
  BinaryClock = new util.Clock();
  Goal_reached = new visual.TextStim({
    win: psychoJS.window,
    name: 'Goal_reached',
    text: "Was the robot able to reach the goal?\n\nenter 'y' for yes and 'n' for no.",
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0], height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  key_resp = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "begin_expt"
  begin_exptClock = new util.Clock();
  text_mid = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_mid',
    text: 'You will now be shown an image of the same robot and environment. Your job is to decide whether the robot will be able to reach the goal and your confidence in this decision.\n\nPress any button to proceed. ',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0], height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  key_resp_mid = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "choose"
  chooseClock = new util.Clock();
  image = new visual.ImageStim({
    win : psychoJS.window,
    name : 'image', units : undefined, 
    image : undefined, mask : undefined,
    ori : 0.0, pos : [0, 0], size : [0.5, 0.5],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : 0.0 
  });
  // Initialize components for Routine "Binary2"
  Binary2Clock = new util.Clock();
  Question = new visual.TextStim({
    win: psychoJS.window,
    name: 'Question',
    text: "Would the robot be able to get to the cone frome here?\n\nenter 'y' for yes and 'n' for no.",
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0], height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  yes_no = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "confidence"
  confidenceClock = new util.Clock();
  text_confidence = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_confidence',
    text: "How confident are you?\n\n'Not confident at all' (press '1')\n'Somewhat not confident' (press '2')\n'Neutral' (press '3')\n'Somewhat confident' (press '4')\n'Very confident' (press '5')",
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0], height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  key_resp_confidence = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "thanks"
  thanksClock = new util.Clock();
  text_thanks = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_thanks',
    text: "That's it. Well done on completing the experiment. \n\nPlease follow this link to get paid: \n\n\n\nAnd thank you very much for your time.",
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0], height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  key_resp_thanks = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Create some handy timers
  globalClock = new util.Clock();  // to track the time since experiment started
  routineTimer = new util.CountdownTimer();  // to track time remaining of each (non-slip) routine
  
  return Scheduler.Event.NEXT;
}


var t;
var frameN;
var continueRoutine;
var _key_resp_inst_allKeys;
var instructionsComponents;
function instructionsRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'instructions' ---
    t = 0;
    instructionsClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    key_resp_inst.keys = undefined;
    key_resp_inst.rt = undefined;
    _key_resp_inst_allKeys = [];
    // keep track of which components have finished
    instructionsComponents = [];
    instructionsComponents.push(text_inst);
    instructionsComponents.push(key_resp_inst);
    
    instructionsComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}


function instructionsRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'instructions' ---
    // get current time
    t = instructionsClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text_inst* updates
    if (t >= 0.0 && text_inst.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_inst.tStart = t;  // (not accounting for frame time here)
      text_inst.frameNStart = frameN;  // exact frame index
      
      text_inst.setAutoDraw(true);
    }

    
    // *key_resp_inst* updates
    if (t >= 0.0 && key_resp_inst.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp_inst.tStart = t;  // (not accounting for frame time here)
      key_resp_inst.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp_inst.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp_inst.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp_inst.clearEvents(); });
    }

    if (key_resp_inst.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp_inst.getKeys({keyList: [], waitRelease: false});
      _key_resp_inst_allKeys = _key_resp_inst_allKeys.concat(theseKeys);
      if (_key_resp_inst_allKeys.length > 0) {
        key_resp_inst.keys = _key_resp_inst_allKeys[_key_resp_inst_allKeys.length - 1].name;  // just the last key pressed
        key_resp_inst.rt = _key_resp_inst_allKeys[_key_resp_inst_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    instructionsComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function instructionsRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'instructions' ---
    instructionsComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(key_resp_inst.corr, level);
    }
    psychoJS.experiment.addData('key_resp_inst.keys', key_resp_inst.keys);
    if (typeof key_resp_inst.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_resp_inst.rt', key_resp_inst.rt);
        routineTimer.reset();
        }
    
    key_resp_inst.stop();
    // the Routine "instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var practice_trials;
function practice_trialsLoopBegin(practice_trialsLoopScheduler, snapshot) {
  return async function() {
    TrialHandler.fromSnapshot(snapshot); // update internal variables (.thisN etc) of the loop
    
    // set up handler to look after randomisation of conditions etc
    practice_trials = new TrialHandler({
      psychoJS: psychoJS,
      nReps: 1, method: TrialHandler.Method.SEQUENTIAL,
      extraInfo: expInfo, originPath: undefined,
      trialList: TrialHandler.importConditions(psychoJS.serverManager, 'vpr_stimuli1.xlsx', '100:102'),
      seed: undefined, name: 'practice_trials'
    });
    psychoJS.experiment.addLoop(practice_trials); // add the loop to the experiment
    currentLoop = practice_trials;  // we're now the current loop
    
    // Schedule all the trials in the trialList:
    practice_trials.forEach(function() {
      snapshot = practice_trials.getSnapshot();
    
      practice_trialsLoopScheduler.add(importConditions(snapshot));
      practice_trialsLoopScheduler.add(fixateRoutineBegin(snapshot));
      practice_trialsLoopScheduler.add(fixateRoutineEachFrame());
      practice_trialsLoopScheduler.add(fixateRoutineEnd(snapshot));
      practice_trialsLoopScheduler.add(videoRoutineBegin(snapshot));
      practice_trialsLoopScheduler.add(videoRoutineEachFrame());
      practice_trialsLoopScheduler.add(videoRoutineEnd(snapshot));
      practice_trialsLoopScheduler.add(BinaryRoutineBegin(snapshot));
      practice_trialsLoopScheduler.add(BinaryRoutineEachFrame());
      practice_trialsLoopScheduler.add(BinaryRoutineEnd(snapshot));
      practice_trialsLoopScheduler.add(practice_trialsLoopEndIteration(practice_trialsLoopScheduler, snapshot));
    });
    
    return Scheduler.Event.NEXT;
  }
}


async function practice_trialsLoopEnd() {
  // terminate loop
  psychoJS.experiment.removeLoop(practice_trials);
  // update the current loop from the ExperimentHandler
  if (psychoJS.experiment._unfinishedLoops.length>0)
    currentLoop = psychoJS.experiment._unfinishedLoops.at(-1);
  else
    currentLoop = psychoJS.experiment;  // so we use addData from the experiment
  return Scheduler.Event.NEXT;
}


function practice_trialsLoopEndIteration(scheduler, snapshot) {
  // ------Prepare for next entry------
  return async function () {
    if (typeof snapshot !== 'undefined') {
      // ------Check if user ended loop early------
      if (snapshot.finished) {
        // Check for and save orphaned data
        if (psychoJS.experiment.isEntryEmpty()) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        scheduler.stop();
      } else {
        psychoJS.experiment.nextEntry(snapshot);
      }
    return Scheduler.Event.NEXT;
    }
  };
}


var trials;
function trialsLoopBegin(trialsLoopScheduler, snapshot) {
  return async function() {
    TrialHandler.fromSnapshot(snapshot); // update internal variables (.thisN etc) of the loop
    
    // set up handler to look after randomisation of conditions etc
    trials = new TrialHandler({
      psychoJS: psychoJS,
      nReps: 1, method: TrialHandler.Method.RANDOM,
      extraInfo: expInfo, originPath: undefined,
      trialList: TrialHandler.importConditions(psychoJS.serverManager, 'vpr_stimuli1.xlsx', '0:10'),
      seed: undefined, name: 'trials'
    });
    psychoJS.experiment.addLoop(trials); // add the loop to the experiment
    currentLoop = trials;  // we're now the current loop
    
    // Schedule all the trials in the trialList:
    trials.forEach(function() {
      snapshot = trials.getSnapshot();
    
      trialsLoopScheduler.add(importConditions(snapshot));
      trialsLoopScheduler.add(fixateRoutineBegin(snapshot));
      trialsLoopScheduler.add(fixateRoutineEachFrame());
      trialsLoopScheduler.add(fixateRoutineEnd(snapshot));
      trialsLoopScheduler.add(chooseRoutineBegin(snapshot));
      trialsLoopScheduler.add(chooseRoutineEachFrame());
      trialsLoopScheduler.add(chooseRoutineEnd(snapshot));
      trialsLoopScheduler.add(Binary2RoutineBegin(snapshot));
      trialsLoopScheduler.add(Binary2RoutineEachFrame());
      trialsLoopScheduler.add(Binary2RoutineEnd(snapshot));
      trialsLoopScheduler.add(confidenceRoutineBegin(snapshot));
      trialsLoopScheduler.add(confidenceRoutineEachFrame());
      trialsLoopScheduler.add(confidenceRoutineEnd(snapshot));
      trialsLoopScheduler.add(trialsLoopEndIteration(trialsLoopScheduler, snapshot));
    });
    
    return Scheduler.Event.NEXT;
  }
}


async function trialsLoopEnd() {
  // terminate loop
  psychoJS.experiment.removeLoop(trials);
  // update the current loop from the ExperimentHandler
  if (psychoJS.experiment._unfinishedLoops.length>0)
    currentLoop = psychoJS.experiment._unfinishedLoops.at(-1);
  else
    currentLoop = psychoJS.experiment;  // so we use addData from the experiment
  return Scheduler.Event.NEXT;
}


function trialsLoopEndIteration(scheduler, snapshot) {
  // ------Prepare for next entry------
  return async function () {
    if (typeof snapshot !== 'undefined') {
      // ------Check if user ended loop early------
      if (snapshot.finished) {
        // Check for and save orphaned data
        if (psychoJS.experiment.isEntryEmpty()) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        scheduler.stop();
      } else {
        psychoJS.experiment.nextEntry(snapshot);
      }
    return Scheduler.Event.NEXT;
    }
  };
}


var fixateComponents;
function fixateRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'fixate' ---
    t = 0;
    fixateClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    routineTimer.add(1.000000);
    // update component parameters for each repeat
    // keep track of which components have finished
    fixateComponents = [];
    fixateComponents.push(fixate_middle);
    fixateComponents.push(text);
    
    fixateComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}


var frameRemains;
function fixateRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'fixate' ---
    // get current time
    t = fixateClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *fixate_middle* updates
    if (t >= 0.5 && fixate_middle.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      fixate_middle.tStart = t;  // (not accounting for frame time here)
      fixate_middle.frameNStart = frameN;  // exact frame index
      
      fixate_middle.setAutoDraw(true);
    }

    frameRemains = 0.5 + 0.5 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (fixate_middle.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      fixate_middle.setAutoDraw(false);
    }
    
    // *text* updates
    if (t >= 0.0 && text.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text.tStart = t;  // (not accounting for frame time here)
      text.frameNStart = frameN;  // exact frame index
      
      text.setAutoDraw(true);
    }

    frameRemains = 0.0 + 0.5 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (text.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      text.setAutoDraw(false);
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    fixateComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function fixateRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'fixate' ---
    fixateComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var movieClock;
var movie;
var videoComponents;
function videoRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'video' ---
    t = 0;
    videoClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    routineTimer.add(7.000000);
    // update component parameters for each repeat
    movieClock = new util.Clock();
    movie = new visual.MovieStim({
      win: psychoJS.window,
      name: 'movie',
      units: undefined,
      movie: video,
      pos: [0, 0],
      size: [0.5, 0.5],
      ori: 0.0,
      opacity: undefined,
      loop: false,
      noAudio: false,
      });
    // keep track of which components have finished
    videoComponents = [];
    videoComponents.push(movie);
    
    videoComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}


function videoRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'video' ---
    // get current time
    t = videoClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *movie* updates
    if (t >= 0.0 && movie.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      movie.tStart = t;  // (not accounting for frame time here)
      movie.frameNStart = frameN;  // exact frame index
      
      movie.setAutoDraw(true);
      movie.play();
    }

    frameRemains = 0.0 + 7.0 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (movie.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      movie.setAutoDraw(false);
    }

    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    videoComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function videoRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'video' ---
    videoComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    movie.stop();
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var _key_resp_allKeys;
var BinaryComponents;
function BinaryRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'Binary' ---
    t = 0;
    BinaryClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    routineTimer.add(5.000000);
    // update component parameters for each repeat
    key_resp.keys = undefined;
    key_resp.rt = undefined;
    _key_resp_allKeys = [];
    // keep track of which components have finished
    BinaryComponents = [];
    BinaryComponents.push(Goal_reached);
    BinaryComponents.push(key_resp);
    
    BinaryComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}


function BinaryRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'Binary' ---
    // get current time
    t = BinaryClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *Goal_reached* updates
    if (t >= 0.0 && Goal_reached.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      Goal_reached.tStart = t;  // (not accounting for frame time here)
      Goal_reached.frameNStart = frameN;  // exact frame index
      
      Goal_reached.setAutoDraw(true);
    }

    frameRemains = 0.0 + 3.0 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (Goal_reached.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      Goal_reached.setAutoDraw(false);
    }
    
    // *key_resp* updates
    if (t >= 0.0 && key_resp.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp.tStart = t;  // (not accounting for frame time here)
      key_resp.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp.clearEvents(); });
    }

    frameRemains = 0.0 + 5.0 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (key_resp.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      key_resp.status = PsychoJS.Status.FINISHED;
  }

    if (key_resp.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp.getKeys({keyList: ['y', 'n'], waitRelease: false});
      _key_resp_allKeys = _key_resp_allKeys.concat(theseKeys);
      if (_key_resp_allKeys.length > 0) {
        key_resp.keys = _key_resp_allKeys[_key_resp_allKeys.length - 1].name;  // just the last key pressed
        key_resp.rt = _key_resp_allKeys[_key_resp_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    BinaryComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function BinaryRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'Binary' ---
    BinaryComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(key_resp.corr, level);
    }
    psychoJS.experiment.addData('key_resp.keys', key_resp.keys);
    if (typeof key_resp.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_resp.rt', key_resp.rt);
        routineTimer.reset();
        }
    
    key_resp.stop();
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var _key_resp_mid_allKeys;
var begin_exptComponents;
function begin_exptRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'begin_expt' ---
    t = 0;
    begin_exptClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    key_resp_mid.keys = undefined;
    key_resp_mid.rt = undefined;
    _key_resp_mid_allKeys = [];
    // keep track of which components have finished
    begin_exptComponents = [];
    begin_exptComponents.push(text_mid);
    begin_exptComponents.push(key_resp_mid);
    
    begin_exptComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}


function begin_exptRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'begin_expt' ---
    // get current time
    t = begin_exptClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text_mid* updates
    if (t >= 0.0 && text_mid.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_mid.tStart = t;  // (not accounting for frame time here)
      text_mid.frameNStart = frameN;  // exact frame index
      
      text_mid.setAutoDraw(true);
    }

    
    // *key_resp_mid* updates
    if (t >= 0.0 && key_resp_mid.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp_mid.tStart = t;  // (not accounting for frame time here)
      key_resp_mid.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp_mid.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp_mid.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp_mid.clearEvents(); });
    }

    if (key_resp_mid.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp_mid.getKeys({keyList: [], waitRelease: false});
      _key_resp_mid_allKeys = _key_resp_mid_allKeys.concat(theseKeys);
      if (_key_resp_mid_allKeys.length > 0) {
        key_resp_mid.keys = _key_resp_mid_allKeys[_key_resp_mid_allKeys.length - 1].name;  // just the last key pressed
        key_resp_mid.rt = _key_resp_mid_allKeys[_key_resp_mid_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    begin_exptComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function begin_exptRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'begin_expt' ---
    begin_exptComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(key_resp_mid.corr, level);
    }
    psychoJS.experiment.addData('key_resp_mid.keys', key_resp_mid.keys);
    if (typeof key_resp_mid.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_resp_mid.rt', key_resp_mid.rt);
        routineTimer.reset();
        }
    
    key_resp_mid.stop();
    // the Routine "begin_expt" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var chooseComponents;
function chooseRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'choose' ---
    t = 0;
    chooseClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    routineTimer.add(2.000000);
    // update component parameters for each repeat
    image.setImage(img);
    // keep track of which components have finished
    chooseComponents = [];
    chooseComponents.push(image);
    
    chooseComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}


function chooseRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'choose' ---
    // get current time
    t = chooseClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *image* updates
    if (t >= 0.0 && image.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      image.tStart = t;  // (not accounting for frame time here)
      image.frameNStart = frameN;  // exact frame index
      
      image.setAutoDraw(true);
    }

    frameRemains = 0.0 + 2 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (image.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      image.setAutoDraw(false);
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    chooseComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function chooseRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'choose' ---
    chooseComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var _yes_no_allKeys;
var Binary2Components;
function Binary2RoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'Binary2' ---
    t = 0;
    Binary2Clock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    routineTimer.add(5.000000);
    // update component parameters for each repeat
    yes_no.keys = undefined;
    yes_no.rt = undefined;
    _yes_no_allKeys = [];
    // keep track of which components have finished
    Binary2Components = [];
    Binary2Components.push(Question);
    Binary2Components.push(yes_no);
    
    Binary2Components.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}


function Binary2RoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'Binary2' ---
    // get current time
    t = Binary2Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *Question* updates
    if (t >= 0.0 && Question.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      Question.tStart = t;  // (not accounting for frame time here)
      Question.frameNStart = frameN;  // exact frame index
      
      Question.setAutoDraw(true);
    }

    frameRemains = 0.0 + 3.0 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (Question.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      Question.setAutoDraw(false);
    }
    
    // *yes_no* updates
    if (t >= 0.0 && yes_no.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      yes_no.tStart = t;  // (not accounting for frame time here)
      yes_no.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { yes_no.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { yes_no.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { yes_no.clearEvents(); });
    }

    frameRemains = 0.0 + 5.0 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (yes_no.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      yes_no.status = PsychoJS.Status.FINISHED;
  }

    if (yes_no.status === PsychoJS.Status.STARTED) {
      let theseKeys = yes_no.getKeys({keyList: ['y', 'n'], waitRelease: false});
      _yes_no_allKeys = _yes_no_allKeys.concat(theseKeys);
      if (_yes_no_allKeys.length > 0) {
        yes_no.keys = _yes_no_allKeys[_yes_no_allKeys.length - 1].name;  // just the last key pressed
        yes_no.rt = _yes_no_allKeys[_yes_no_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    Binary2Components.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function Binary2RoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'Binary2' ---
    Binary2Components.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(yes_no.corr, level);
    }
    psychoJS.experiment.addData('yes_no.keys', yes_no.keys);
    if (typeof yes_no.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('yes_no.rt', yes_no.rt);
        routineTimer.reset();
        }
    
    yes_no.stop();
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var _key_resp_confidence_allKeys;
var confidenceComponents;
function confidenceRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'confidence' ---
    t = 0;
    confidenceClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    key_resp_confidence.keys = undefined;
    key_resp_confidence.rt = undefined;
    _key_resp_confidence_allKeys = [];
    // keep track of which components have finished
    confidenceComponents = [];
    confidenceComponents.push(text_confidence);
    confidenceComponents.push(key_resp_confidence);
    
    confidenceComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}


function confidenceRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'confidence' ---
    // get current time
    t = confidenceClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text_confidence* updates
    if (t >= 0.0 && text_confidence.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_confidence.tStart = t;  // (not accounting for frame time here)
      text_confidence.frameNStart = frameN;  // exact frame index
      
      text_confidence.setAutoDraw(true);
    }

    
    // *key_resp_confidence* updates
    if (t >= 0.0 && key_resp_confidence.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp_confidence.tStart = t;  // (not accounting for frame time here)
      key_resp_confidence.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp_confidence.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp_confidence.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp_confidence.clearEvents(); });
    }

    if (key_resp_confidence.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp_confidence.getKeys({keyList: ['1', '2', '3', '4', '5'], waitRelease: false});
      _key_resp_confidence_allKeys = _key_resp_confidence_allKeys.concat(theseKeys);
      if (_key_resp_confidence_allKeys.length > 0) {
        key_resp_confidence.keys = _key_resp_confidence_allKeys[_key_resp_confidence_allKeys.length - 1].name;  // just the last key pressed
        key_resp_confidence.rt = _key_resp_confidence_allKeys[_key_resp_confidence_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    confidenceComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function confidenceRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'confidence' ---
    confidenceComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(key_resp_confidence.corr, level);
    }
    psychoJS.experiment.addData('key_resp_confidence.keys', key_resp_confidence.keys);
    if (typeof key_resp_confidence.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_resp_confidence.rt', key_resp_confidence.rt);
        routineTimer.reset();
        }
    
    key_resp_confidence.stop();
    // the Routine "confidence" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var _key_resp_thanks_allKeys;
var thanksComponents;
function thanksRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'thanks' ---
    t = 0;
    thanksClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    key_resp_thanks.keys = undefined;
    key_resp_thanks.rt = undefined;
    _key_resp_thanks_allKeys = [];
    // keep track of which components have finished
    thanksComponents = [];
    thanksComponents.push(text_thanks);
    thanksComponents.push(key_resp_thanks);
    
    thanksComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}


function thanksRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'thanks' ---
    // get current time
    t = thanksClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text_thanks* updates
    if (t >= 0.0 && text_thanks.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_thanks.tStart = t;  // (not accounting for frame time here)
      text_thanks.frameNStart = frameN;  // exact frame index
      
      text_thanks.setAutoDraw(true);
    }

    
    // *key_resp_thanks* updates
    if (t >= 0.0 && key_resp_thanks.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp_thanks.tStart = t;  // (not accounting for frame time here)
      key_resp_thanks.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp_thanks.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp_thanks.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp_thanks.clearEvents(); });
    }

    if (key_resp_thanks.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp_thanks.getKeys({keyList: [], waitRelease: false});
      _key_resp_thanks_allKeys = _key_resp_thanks_allKeys.concat(theseKeys);
      if (_key_resp_thanks_allKeys.length > 0) {
        key_resp_thanks.keys = _key_resp_thanks_allKeys[_key_resp_thanks_allKeys.length - 1].name;  // just the last key pressed
        key_resp_thanks.rt = _key_resp_thanks_allKeys[_key_resp_thanks_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    thanksComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function thanksRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'thanks' ---
    thanksComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(key_resp_thanks.corr, level);
    }
    psychoJS.experiment.addData('key_resp_thanks.keys', key_resp_thanks.keys);
    if (typeof key_resp_thanks.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_resp_thanks.rt', key_resp_thanks.rt);
        routineTimer.reset();
        }
    
    key_resp_thanks.stop();
    // the Routine "thanks" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


function importConditions(currentLoop) {
  return async function () {
    psychoJS.importAttributes(currentLoop.getCurrentTrial());
    return Scheduler.Event.NEXT;
    };
}


async function quitPsychoJS(message, isCompleted) {
  // Check for and save orphaned data
  if (psychoJS.experiment.isEntryEmpty()) {
    psychoJS.experiment.nextEntry();
  }
  psychoJS.window.close();
  psychoJS.quit({message: message, isCompleted: isCompleted});
  
  return Scheduler.Event.QUIT;
}
