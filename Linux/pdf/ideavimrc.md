    set scrolloff=5

    " Don't use Ex mode, use Q for formatting.
    map Q gq

    " --- Enable IdeaVim plugins https://jb.gg/ideavim-plugins

    " Vim 的默认寄存器和系统剪贴板共享
    set clipboard+=unnamed
    set history=100000

    " select模式下复制
    if has("clipboard")
        vnoremap <C-C> "+y
    endif

    " 不用方向键进行移动
    nnoremap <Up> <Nop>
    nnoremap <Down> <Nop>
    nnoremap <Left> <Nop>
    nnoremap <Right> <Nop>
    let mapleader=' '

    " ==================================================
    " Show all the provided actions via `:actionlist`
    " ==================================================
    " project search

    nnoremap <Leader>fg :action SearchEverywhere<CR>
    nnoremap <Leader>ff :action ReformatCode<CR>
    nnoremap <Leader>fi :action CollapseAllRegions<CR>
    nnoremap <Leader>fo :action ExpandAllRegions<CR>
    nnoremap <Leader>fn :action Generate<CR>
    nnoremap <Leader>fb :action GotoImplementation<CR>
    nnoremap <Leader>fc :action CommentByLineComment<CR>
    nnoremap <Leader>fr :action RecentFiles<CR>
    nnoremap <Leader>fj :action FindInPath<CR>
    nnoremap <Leader>fk :action ShowUsages<CR>




    nnoremap <Leader>wn :action Back<CR>
    nnoremap <Leader>wg :action HideAllWindows<CR>
    nnoremap <Leader>wh :action SelectInProjectView<CR>
    nnoremap <Leader>wt :action ActivateTerminalToolWindow<CR>
    nnoremap <Leader>wi :action EditorLineStart<CR>
    nnoremap <Leader>wo :action EditorLineEnd<CR>
    nnoremap <Leader>wb :action GotoDeclaration<CR>

