"General items
set number
set showcmd
set ruler
set encoding=utf-8
set autoindent
set smartindent
set listchars=tab:>>\ ,trail:*
noremap <C-p> :w <cr> :!python3 %<cr>

"Plugins

call plug#begin('~/.local/share/nvim/plugged')
		"curl -fLo ~/.local/share/nvim/site/autoload/plug.vim --create-dirs \
		"https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
		"Sample: jdhao.github.io/2018/12/24/centos_nvim_install_use_guide_en/
"Functions
	"Auto completion
"Plug 'davidhalter/jedi-vim'
"Plug 'Shougo/deoplete.nvim', { 'do': ':UpdateRemotePlugins' }
"let g:deoplete#enable_at_startup = 1
"Plug 'zchee/deoplete-jedi'
	"Auto quote and bracket completion
"Plug 'jiangmiao/auto-pairs'
	" Code auto-format plugin
"Plug 'sbdchd/neoformat'
Plug 'lervag/vimtex'
	" Enable alignment
let g:neoformat_basic_format_align = 1 
	" Enable tab to space conversion
let g:neoformat_basic_format_retab = 1
	" Enable trimming of trailing whitespace
let g:neoformat_basic_format_trim = 1
	"File management
Plug 'scrooloose/nerdtree'
	"Code checker
Plug 'neomake/neomake'
let g:neomake_python_enabled_makers = ['pylint']
	"Code folding
Plug 'tmhedberg/SimpylFold'

"Themes
Plug 'morhetz/gruvbox'

call plug#end()

filetype plugin indent on
syntax enable

let g:vimtex_view_method = 'zathura'
colorscheme gruvbox
set background=dark
