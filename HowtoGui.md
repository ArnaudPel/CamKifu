#HOW-TO : GUI

This tutorial is an insight on how the GUI code has been structured so far. Recognizing the great amount of intelligence inside the coding community, I recognize that all this can most likely be improved, made more graceful. So, again, any improvement suggestion or work is of course welcome with humility. Especially in the area of setting method references from outside (proxy kinda thing)


## Communication GUI -> Video Processor

This part is about passing information from the GUI to the workers. In this example, the objective is to implement a functionality that would let a developer make a Video Processor wait for his signal before processing the next frame. This can come in handy when having noticed a particular case requiring step-by-step visual analysis. Let's add a button for that.


### 1. class VUI
Since our new feature is related to computer vision, the `VUI` class is the preferred location to add the button. Let's mimick what's been done for other buttons, and edit the `init_components()` method:

```python
def init_components(self):
    ...
    b_next = Button(self.buttons, text="Next", command=lambda: self.execute("next"))
    b_next.grid(row=5, column=0, columnspan=2)  # add to a new row on the screen
```

What's that `execute("next")` ? A loose method call that will look for the `"next"` key in `ui.commands` and execute it if present. So let's create that method and link it ! Two middlemen will be needed for the GUI to interact with the Video Processors : `ControllerV` and `VManager`.

### 2. class ControllerV
`ControllerV` holds the reference to the GUI instance, and sees it as an `input`. The `__init__` method already links some commands, let's add a similar line.

```python
def __init__(self, user_input, display, sgffile=None, video=0, bounds=(0, 1)):
    ...
    self.input.commands["next"] = lambda: self.next()  # lambda needed to bind method externally at runtime
    ...
```

That `next` method doesn't exist yet, and it is to be set externally by the second middleman. For that reason, a lambda mechanism is used to point dynamically to the method. No more code is actually needed in that class, yet in order to be a bit more explicit, let's add a "doc" method giving a hint on what's going on.

```python
def next(self):
    """
    To be set from outside (eg. by Vision Manager).
    The user has clicked "Next".

    """
    pass
```

### 3. class VManager
`VManager` is responsible for managing the vision threads and their interactions. Hence it holds the references to the Video Processors that we'd like to talk to. That's where we can implement the empty `next` method of `ControllerV`. 
Since our functionality also makes sense in dev (single-threaded) mode, so let's go crazy and edit `VManagerBase` directly.
 
```python
 class VManagerBase(Thread):
    ...
    def next(self):
        """
        Call next() on all VidProcessor.
        
        """
        # "not None" check ommitted for concision's sake 
        self.board_finder.next()
        self.stones_finder.next()
```

Now we can link that implementation to the controller via the `_bind_controller` method. 
 ```python
 class VManagerBase(Thread):
    ...
    def bind_controller(self):
       ...
       self.controller.next = self.next
 ```

Good. Now, since `VManager` happens to have a more dynamic structure listing its processors, let's take a sec to override our newly created `next()` method on the way:

```python
 class VManager(VManagerBase):
    ...
    def next(self):
        for proc in self.processes:
            proc.next()
```

### 4. class VidProcessor
It is now time to implement the handling of our signal by the worker, by means of a flag variable.

```python
def __init__(self, vmanager):
    ...
    self.next_flag = False  # see self.next()
    ...
    
def next(self):
    """
    Indicate that one frame may be allowed to be read if self is in "paused" state. Has no effect if self is 
    not paused, since in this case frames are supposed to be flowing already. 
    
    """
    self.next_flag = True

# the implementation of the "next" mechanism itself is not the point here, but let's show it for completion's sake
def _checkpause(self):
    ...
    if self.vmanager.imqueue is not None:  # supposedly a multi-threaded env
        while self.pausedflag and not self.next_flag:
            sleep(0.1)
        self.next_flag = False
```

Testing... Righteo, seems to be working as expected ! Now let's hope we haven't introduced too many bugs with that 
change :)

### 5. final goody
How about adding a keyboard shortcut while we're at it ? My mouse has been tending to double-click on it's own 
freewill lately, so that may come in handy. Let's go back to the start, with `VUI`:

```python
    from golib.gui.ui import mod1
    ...
    def init_components(self):
        ...
        self.bind_all("<{0}-f>".format(mod1), lambda _: self.execute("next"))
```

There we go !

I hope this example helps understanding the general picture of the one-sided `GUI -> VidProcessor` communication. 
