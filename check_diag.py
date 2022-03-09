#!/usr/bin/python3

# Check for the existence of diagnostic DIAGNOSTIC in esperiment EXPERIMENT
# for shots in FILE

# Pipe it with less for tidier std out!

# made by:
# - Luca Cinnirella
#
# and modified/updated by:
#
# Last update: 09.03.2022

# IMPORT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import sys
from optparse import OptionParser, OptionGroup
import os
import aug_sfutils as sf



# OPTIONS HANDLER
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
parser = OptionParser(usage='Usage: %prog [mandatory] args [options] args\n' +
                            '       Check for the existence of diagnostic ' +
                                    'DIAGNOSTIC in esperiment EXPERIMENT ' +
                                    'for shots in FILE\n' +
                            '       Pipe it with less for tidier std out!',
                      add_help_option=False)
parser.add_option('-h', '--help',
                  action='help',
                  help='Show this help message')
parser.add_option('-e', '--experiment',
                  metavar='EXPERIMENT',
                  action='store', type='str', dest='exp', default='AUGD',
                  help='Select the name of the experiment')
mandatory = OptionGroup(parser,
                        'Mandatory args',
                        'These args are mandatory!!!')
mandatory.add_option('-f', '--file',
                     metavar='FILE',
                     action='store', type='str', dest='file',
                     help='Select file from which shot numbers are to be ' +
                          'read (it cans also be a comma separated list)')
mandatory.add_option('-d', '--diagnostic',
                     metavar='DIAGNOSTIC',
                     action='store', type='str', dest='diag',
                     help='Check the selected diagnostics (single, comma ' +
                          'separated or file) data existence for each shot ' +
                          'in FILE')
parser.add_option_group(mandatory)
(options, args) = parser.parse_args()
if not(options.file and options.diag):
    parser.print_help()
    sys.exit('No file or no diagnostic was provided')
experiment = options.exp

if os.path.exists(options.file):
    with open(options.file) as f:
        shots = f.read().splitlines()
else:
    shots = options.file.split(',')

if os.path.exists(options.diag):
    with open(options.diag) as f:
        diagnostics = f.read().splitlines()
else:
    diagnostics = options.diag.split(',')




# QUERYING SHOTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get shot-files relative to Function Parametrization and equilibrium
for shot in shots:
    shot = int(shot)

    print(f'For experiment {experiment} and shot {shot}')
    for diagnostic in diagnostics:
        try:
            diag_test = sf.SFREAD(shot, diagnostic, exp=experiment)
            if diag_test.status:
                print(f'diagnostic {diagnostic} exists')
            else:
                sys.exit('Error while laoding ' + diagnostic)
        except:
            print(f'diagnostic {diagnostic} does NOT exist')
    print('\n')



print('\nPipe it with less for tidier std out!')
