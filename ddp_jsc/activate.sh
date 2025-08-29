#!/usr/bin/env bash

_curr_file="${BASH_SOURCE[0]:-${(%):-%x}}"
_curr_dir="$(dirname "$_curr_file")"

venv_dir="$_curr_dir"/env-"$SYSTEMNAME"

[ "$(command -v deactivate)" ] && deactivate

module --force purge
if ! [ -f "$_curr_dir"/modules.sh ]; then
    echo "Cannot find \`$_curr_dir/modules.sh\`; its existence is required."
    return 1
fi
source "$_curr_dir"/modules.sh

if ! [ -d "$venv_dir" ]; then
    echo "Please manually execute \`nice bash ${_curr_dir@Q}/set_up.sh\`" \
         "on a login node to create the \`venv\`."
    return 1
elif ! [ -f "$venv_dir"/bin/activate ]; then
    echo "Something seems to be wrong with the \`venv\` at \`$venv_dir\`." \
         "Please delete it (\`nice rm -rf ${venv_dir@Q}\`) and" \
         "execute \`nice bash ${_curr_dir@Q}/set_up.sh\`" \
         "on a login node to re-create the \`venv\`."
    return 1
fi
source "$venv_dir"/bin/activate
