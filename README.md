# Utilities pack for ASDEX-Upgrade

## Introduction and Purposes

This repo provides all the code that I developed and modified while I was working at IPP in Garching and, more generally, on the ASDEX-Upgrade tokamak.

## Setting up

Since the script has been written and tested with `Python 3.6`, running it with the aforementioned Python version is highly recommended. I've also tried running it with `Python 3.8` but some fixes, which will come anytime soon, are needed.\
`Python2` won't work for sure!!!

### Bash script installation

***Working on it***

### Manual installation

To manually install all the dependecies just use
```bash
pip install numpy matplotlib pandas scipy aug_sfutils
```
if your default version is a `Python3` version, otherwise you have to
```bash
pip3 install numpy matplotlib pandas scipy aug_sfutils
```
You can also do
```bash
python3.6 -m pip install numpy matplotlib pandas scipy aug_sfutils
```
if neither `Python` nor `Python3` is `Python3.6`

## Running

First of all give execution permission to all the codes that are meant to be executed (all the other files are modules to be imported or guides):
```bash
chmod +x XA.py XPR.py check_diag.py
```
Then, just doing `./script_to_be_executed.py -s SHOT_NUMBER` should work for most of them. However, I recommend to first give a look at each script specific instructions which can be queried by not giving any argument to the script (`./script_to_be_executed.py`), by giving it the `help` flag (`./script_to_be_executed.py -h` or `./script_to_be_executed.py --help`), or just by jumping to the [Guides](#guides) section.

**Warning!**\
If your `Python3` isn't `Python3.6` and executing the scripts still doesn't work, you can try to manually modify each Shebang (change `#!/usr/bin/python3` to `#!/usr/bin/python3.6`) or you can force your shell to ignore it by executing
```bash
python3.6 script_to_be_executed.py -s SHOT_NUMBER
```

## Guides

### XPR.py

This utility displays all the data relative to X-Point Radiator (XPR) detection and tracking. It includes:

- temperature and density relative to 10th, 11th and 12th cores from the divertor Thompson scattering system
- pressure evaluated nearby the X-point via the aforementioned time traces
- radiation time traces and evolving profiles from the DLX and the DDC bolometers
- a poloidal projection displaying the evolution of the XPR detected position

The XPR detection process starts with a median filter being applied over the bolometers time traces. The default window length is 50 ms but it can be changed through the option `-w` or `--window` as
```bash
./XPR.py -s SHOT_NUMBER -w TIME_WINDOW
```
or
```bash
./XPR.py -s SHOT_NUMBER --window=TIME_WINDOW
```
where `TIME_WINDOW` is the window length itself expressed in milliseconds. The step size of the filter, instead, is fixed at 1/3 of the window length, so that the code can still run relatively quickly with at least a small amount of overlapping between different time windows.\
Once the median filter has been applied, the resulting signals are piped to a function which uses `scipy.singal.find_peaks` to return the discrete position of relative maxima in terms of bolometers line of sight. Subsequently, a smoothing operation is applied by performing a weighted average, or a non-deterministic gaussian fit if the `-g` (`--gaussian_fit`) flag is set, on the five points centered around the XPR discrete position.
Finally, the code carries out a transformation, described [here](./Bol_coord_notes.pdf), from line of sight coordinates to real space coordinates.

Output of `./XPR.py -h`:
```
Usage: XPR.py [mandatory] args [options] arg

Options:
  -h, --help            Show this help message
  -w TIME_WINDOW, --window=TIME_WINDOW
                        Set the time window (in ms) length on which to apply
                        median filter. The step size of the median filter is
                        also set to 1/3 of the given window length. In this
                        way a decimation is also applied together with the
                        filter (default is 50)
  -e EQUILIBRIUM_DIAGNOSTIC, --equilibrium_diagnostic=EQUILIBRIUM_DIAGNOSTIC
                        Select which diagnostic is used for magnetic
                        reconstruction (FPP, EQI or EQH, default is EQH)
  -t END_time, --end_time=END_time
                        Select the upper boundary of time interval to plot
  -d CONTOUR_DEPTH, --depth=CONTOUR_DEPTH
                        Set matplotlib contourf depth to CONTOUR_DEPTH
                        (default is 50)
  -f FRAMES_PER_SECOND, --frames-per-second=FRAMES_PER_SECOND
                        Set matplotlib animation fps (default is 30)
  -l, --linear_detrend  Use linear detrend from scipy in order to remove the
                        baseline radiation from DDC bolometer (non-
                        deterministic) DEPRECATED!!!
  -g, --gaussian_fit    Use gaussian fit from scipy in order to get the non-
                        discrete X-point radiator position (non-deterministic)
  -p XPR_START_TIME, --print_to_csv=XPR_START_TIME
                        Output a XPR_position.csv file containing data about
                        the x-point radiator position, the time base and a
                        variable that tells whether or not the xpr is already
                        established

  Mandatory args:
    These args are mandatory!!!

    -s SHOT_NUMBER, --shot=SHOT_NUMBER
                        Execute the code for shot #SHOT_NUMBER
```
If you have more questions you can directly read the [code](./XPR.py) yourself (it should be well commented) or, if this still doesn't work for you, just e-mail [me](mailto:luca.cinnirella@tutanota.com)! :)

### XA.py

***Still under development***

### check_diag.py

This utility checks if shotfiles related to a particular diagnostic set were written during all the shots listed in a newline separated ASCII file or a comma separated list. The output is displayed directly in the terminal and, since it may be polluted with messages from the `aug_sfutils` module, I strongly suggest piping it to `less`:
```bash
./check_diag.py -f FILE -d DIAGNOSTIC | less
```
If the output it's still a bit messy, just press any vi movement key (`h`, `j`, `k` or `l`) and it should become cleaner.

The only non-mandatory option that can be provided is `-e` (`--experiment=EXPERIMENT`), which allows you to select a different experiment from AUGD (default).

Output of `./check_diag.py -h`:
```
Usage: check_diag.py [mandatory] args [options] args
       Check for the existence of diagnostic DIAGNOSTIC in esperiment EXPERIMENT for shots in FILE
       Pipe it with less for tidier std out!

Options:
  -h, --help            Show this help message
  -e EXPERIMENT, --experiment=EXPERIMENT
                        Select the name of the experiment

  Mandatory args:
    These args are mandatory!!!

    -f FILE, --file=FILE
                        Select file from which shot numbers are to be read (it
                        cans also be a comma separated list)
    -d DIAGNOSTIC, --diagnostic=DIAGNOSTIC
                        Check the selected diagnostics (single, comma
                        separated or file) data existence for each shot in
                        FILE
```
If you have more questions you can directly read the [code](./check_diag.py) yourself (it should be short and easy to understand) or, if this still doesn't work for you, just e-mail [me](mailto:luca.cinnirella@tutanota.com)! :)

## Other info

### Authors

- [**Luca Cinnirella**](mailto:luca.cinnirella@tutanota.com) - *creator* - [Kruayd (yep, it's me)](https://github.com/Kruayd)
