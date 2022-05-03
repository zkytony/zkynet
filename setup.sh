#!/bin/bash
########### This section is almost fixed for every project ##########
# Run this script from repository root
if [[ ! $PWD = *zkynet ]]; then
    echo "You must be in the root directory of the cos-pomdp repository."
    return 1
fi
repo_root=$PWD

# Utility functions
function first_time_setup
{
    if [ ! -e "$1/.DONE_SETUP" ]; then
        # has not successfully setup
        true && return
    else
        false
    fi
}
#####################################################################

############ sourcing virtualenv business ###########################
if [ ! -d "venv/zkynet" ]; then
    virtualenv -p python3 venv/zkynet
    source venv/zkynet/bin/activate
fi

source_venv=true
if [[ "$VIRTUAL_ENV" == *"zkynet"* ]]; then
    source_venv=false
fi

if [ $source_venv = true ]; then
    source venv/zkynet/bin/activate
fi

# ask if want to create alias command
if [[ $source_venv = false ]]; then
    read -p "Create alias 'zkynet' for starting zkynet venv? " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]];
    then
        echo -e "alias zkynet='cd $repo_root; source setup.bash'" >> ~/.bashrc
    fi
fi
####################################################################
