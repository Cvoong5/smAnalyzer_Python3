set number
set backspace=indent,eol,start
syntax on
highlight pythonRepeat ctermfg = red
highlight pythonKeyword ctermfg = red
highlight pythonStatement ctermfg = red
highlight pythonConditional ctermfg = red
highlight pythonException ctermfg = red
highlight pythonFunction ctermfg = red
highlight pythonOperator ctermfg = red
highlight pythonString ctermfg = yellow
highlight Error ctermfg=blue ctermbg = NONE
highlight Folded guifg=white ctermfg=white·
highlight Folded guibg=NONE ctermbg=NONE

set tabstop=4       " Number of spaces tabs count for
set softtabstop=4   " Number of spaces inserted when pressing tab
set shiftwidth=4    " Number of spaces to use for auto-indentation
set title

set showmode
set hlsearch
set incsearch
set ignorecase
set smartcase
set statusline=%r\ [Line=%l]
set list
set listchars=tab:»\ ,trail:·
set autoindent
set smartindent
set encoding=utf-8
set foldenable
set foldmethod=syntax
set ruler
set showcmd

nnoremap <C-p> :w<CR>:!python3 %<CR>
nnoremap <C-l> gt
nnoremap <C-h> gT·
