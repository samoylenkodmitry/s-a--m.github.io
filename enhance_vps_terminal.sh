#!/bin/bash

# VPS Terminal Enhancement Script
# This script will upload and run the enhancement script on your VPS

set -e

# VPS Configuration
VPS_HOST="159.69.33.249"
VPS_USER="s"
SSH_KEY="~/.ssh/hetz"

echo "🚀 VPS Terminal Enhancement Script"
echo "=================================="
echo "This will enhance your Ubuntu VPS terminal with:"
echo "• Zsh + Oh My Zsh + plugins"
echo "• Starship prompt"
echo "• Modern CLI tools (bat, eza, fzf, etc.)"
echo "• Better tmux configuration"
echo "• Node.js LTS"
echo ""

read -p "Continue? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo "📡 Creating enhancement script..."

# Create the enhancement script that will run on the VPS
cat > /tmp/vps_enhance.sh << 'EOF'
#!/bin/bash

set -e

echo "🎨 Starting VPS Terminal Enhancement..."
echo "====================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo "📦 Step 1: Updating system and installing essential packages..."
sudo apt update
sudo apt install -y zsh curl wget git unzip build-essential

echo "🔧 Step 2: Installing modern CLI tools..."
sudo apt install -y bat eza fd-find ripgrep fzf tree htop neofetch

echo "🐚 Step 3: Installing Oh My Zsh..."
if [ ! -d "$HOME/.oh-my-zsh" ]; then
    RUNZSH=no sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
    echo "✅ Oh My Zsh installed"
else
    echo "⚠️  Oh My Zsh already installed, skipping..."
fi

echo "⭐ Step 4: Installing Starship prompt..."
if ! command_exists starship; then
    curl -sS https://starship.rs/install.sh | sh -s -- -y
    echo "✅ Starship installed"
else
    echo "⚠️  Starship already installed, skipping..."
fi

echo "🔌 Step 5: Installing Zsh plugins..."

# Create custom plugins directory if it doesn't exist
mkdir -p ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/plugins

# zsh-autosuggestions
if [ ! -d "${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/plugins/zsh-autosuggestions" ]; then
    git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
    echo "✅ zsh-autosuggestions installed"
fi

# zsh-syntax-highlighting
if [ ! -d "${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting" ]; then
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
    echo "✅ zsh-syntax-highlighting installed"
fi

# zsh-completions
if [ ! -d "${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/plugins/zsh-completions" ]; then
    git clone https://github.com/zsh-users/zsh-completions ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/plugins/zsh-completions
    echo "✅ zsh-completions installed"
fi

echo "⚙️  Step 6: Configuring Zsh..."
cat > ~/.zshrc << 'ZSHRC_EOF'
# Oh My Zsh configuration
export ZSH="$HOME/.oh-my-zsh"
ZSH_THEME=""  # Using Starship instead

# Plugins
plugins=(
    git
    docker
    kubectl
    zsh-autosuggestions
    zsh-syntax-highlighting
    zsh-completions
    fzf
    tmux
    sudo
    history
    colorize
    colored-man-pages
)

source $ZSH/oh-my-zsh.sh

# Initialize Starship
eval "$(starship init zsh)"

# Modern tool aliases
alias ll='eza -la --git --header --group'
alias ls='eza'
alias la='eza -la --git'
alias lt='eza --tree'
alias cat='batcat'
alias find='fd'
alias grep='rg'
alias top='htop'

# Navigation aliases
alias ..='cd ..'
alias ...='cd ../..'
alias ....='cd ../../..'

# System aliases
alias myip='curl -s ipinfo.io/ip'
alias ports='netstat -tulanp'
alias weather='curl wttr.in'
alias sysinfo='neofetch'
alias diskspace='df -h'
alias meminfo='free -h'

# Git shortcuts
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git log --oneline --graph'
alias gd='git diff'

# Enhanced history
HISTSIZE=50000
SAVEHIST=50000
setopt appendhistory
setopt sharehistory
setopt incappendhistory
setopt histignorealldups
setopt histfindnodups
setopt histignorespace

# Better completion
autoload -U compinit && compinit
zstyle ':completion:*' matcher-list 'm:{a-z}={A-Za-z}'
zstyle ':completion:*' list-colors "${(s.:.)LS_COLORS}"
zstyle ':completion:*' menu select

# Clean startup (no welcome message)
ZSHRC_EOF

echo "✨ Step 7: Configuring Starship prompt..."
mkdir -p ~/.config
cat > ~/.config/starship.toml << 'STARSHIP_EOF'
format = "$directory$git_branch$git_status$character"

[character]
success_symbol = "[➜](bold green)"
error_symbol = "[➜](bold red)"

[directory]
style = "blue"
truncation_length = 3

[git_branch]
symbol = "🌱"
style = "purple"

[git_status]
style = "red"
STARSHIP_EOF

echo "📱 Step 8: Configuring tmux..."
cat > ~/.tmux.conf << 'TMUX_EOF'
# Set prefix to Ctrl-a
set -g prefix C-a
unbind C-b
bind C-a send-prefix

# Split panes with | and -
bind | split-window -h
bind - split-window -v
unbind '"'
unbind %

# Reload config with r
bind r source-file ~/.tmux.conf \; display-message "Config reloaded!"

# Switch panes using Alt-arrow without prefix
bind -n M-Left select-pane -L
bind -n M-Right select-pane -R
bind -n M-Up select-pane -U
bind -n M-Down select-pane -D

# Mouse support
set -g mouse on

# Start at 1
set -g base-index 1
setw -g pane-base-index 1

# Renumber windows
set -g renumber-windows on

# History
set -g history-limit 10000

# Colors
set -g default-terminal "screen-256color"
set -ga terminal-overrides ",*256col*:Tc"

# Status bar
set -g status-bg black
set -g status-fg white
set -g status-interval 60
set -g status-left-length 30
set -g status-left '#[fg=green](#S) #(whoami) '
set -g status-right '#[fg=yellow]#(cut -d " " -f 1-3 /proc/loadavg) #[fg=white]%H:%M'
TMUX_EOF

echo "🔄 Step 9: Installing Node.js..."
if ! command_exists node; then
    curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
    sudo apt-get install -y nodejs
    echo "✅ Node.js installed"
else
    echo "⚠️  Node.js already installed"
fi

echo "🐚 Step 10: Setting Zsh as default shell..."
current_shell=$(echo $SHELL)
zsh_path=$(which zsh)
if [ "$current_shell" != "$zsh_path" ]; then
    chsh -s $zsh_path
    echo "✅ Zsh set as default shell"
else
    echo "⚠️  Zsh already default shell"
fi

echo ""
echo "🎉 =============================================== 🎉"
echo "   Terminal Enhancement Complete!"
echo "🎉 =============================================== 🎉"
echo ""
echo "🔧 What's new:"
echo "  ✅ Zsh with Oh My Zsh framework"
echo "  ✅ Starship prompt with git integration"
echo "  ✅ Auto-suggestions and syntax highlighting"
echo "  ✅ Modern CLI tools: eza, bat, ripgrep, fd, fzf"
echo "  ✅ Enhanced tmux configuration"
echo "  ✅ Node.js LTS"
echo ""
echo "🚀 New commands to try:"
echo "  • ll, la, lt (enhanced ls)"
echo "  • batcat file.txt (syntax highlighting)"
echo "  • rg pattern (better grep)"
echo "  • fd filename (better find)"
echo "  • sysinfo (system info)"
echo "  • weather (current weather)"
echo ""
echo "⚡ To activate everything now:"
echo "   exec zsh"
echo ""
echo "🎨 Your terminal is now supercharged! 🚀"
EOF

echo "📤 Uploading script to VPS..."
scp -i "$SSH_KEY" /tmp/vps_enhance.sh "$VPS_USER@$VPS_HOST:/tmp/"

echo "🚀 Running enhancement script on VPS..."
echo "💡 You'll be prompted for your sudo password..."
ssh -t -i "$SSH_KEY" "$VPS_USER@$VPS_HOST" "chmod +x /tmp/vps_enhance.sh && /tmp/vps_enhance.sh"

echo ""
echo "🎉 VPS Enhancement Complete!"
echo "=========================="
echo ""
echo "🔄 To connect with your enhanced terminal:"
echo "   ssh -i ~/.ssh/hetz s@159.69.33.249"
echo ""
echo "🎯 To run this enhancement again:"
echo "   ./enhance_vps_terminal.sh"
echo ""
echo "✨ Your VPS terminal is now awesome! 🚀"

# Cleanup
rm -f /tmp/vps_enhance.sh
