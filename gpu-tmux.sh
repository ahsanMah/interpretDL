tmux new-session \; \
  send-keys 'conda activate condatensor' C-m \; \
  split-window -h \; \
  send-keys 'nvtop' C-m \; \
  split-window -v \; \
  send-keys 'watch -n2 sensors' C-m \; \
  split-window -h \; \
  send-keys 'watch -n1 "cat /proc/cpuinfo | grep \"^[c]pu MHz\""' C-m \; \
  select-pane -t 0 \; \
  split-window -v \; \
  send-keys 'htop' C-m \; 
