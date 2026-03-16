% posterabstract.cls
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{abstract}

% Base
\LoadClass[11pt,a4paper]{article}

% Required packages
\RequirePackage{geometry}
\RequirePackage{amsmath}
\RequirePackage{graphicx}
\RequirePackage{titlesec}
\RequirePackage{setspace}
\RequirePackage{booktabs}
\RequirePackage{tabularx}
\RequirePackage{xcolor}
\RequirePackage{subcaption}
\RequirePackage{footmisc}
\RequirePackage{afterpage}
\RequirePackage{etoolbox}

% Set margins
\geometry{top=2cm, bottom=2cm, left=2cm, right=2cm}
\pagestyle{empty}

% Font to Times New Roman
\renewcommand{\rmdefault}{ptm}
\renewcommand{\sfdefault}{phv}

% Storage for keywords
\newcommand{\@keywords}{}
\newcommand{\keywords}[1]{\renewcommand{\@keywords}{#1}}

% Title formatting (LEFT JUSTIFIED, condensed author block)
\renewcommand{\maketitle}{
    \noindent
    {\LARGE \textbf{\uppercase{\@title}}} \\[0.5\baselineskip]
    {\small \@author} \\[0.25\baselineskip]
}

% Command to print category and keywords
\newcommand{\printcategorykeywords}{%
    \vspace{1em}%
    {\small
        \textbf{Keywords:} \@keywords
    }%
    \vspace{1em}%
}

% Flag to track if we are in abstract section
\newif\if@inabstract
\@inabstractfalse

% Store original section command
\let\@originalsection\section

% Redefine section command
\renewcommand{\section}[1]{%
    % If we were in abstract, print category and keywords first
    \if@inabstract
        \printcategorykeywords
        \@inabstractfalse
    \fi
    % Check if this is the Abstract section
    \def\@tempa{#1}%
    \def\@tempb{Abstract}%
    \ifx\@tempa\@tempb
        \@inabstracttrue
    \fi
    % Call original section command
    \@originalsection{#1}%
}

% Ensure category and keywords print at end of document if abstract is last
\AtEndDocument{%
    \if@inabstract
        \printcategorykeywords
        \@inabstractfalse
    \fi
}

% Section heading formatting (larger, reduced space above)
\titleformat{\section}[block]{\normalfont\Large\bfseries}{}{0em}{}
\titlespacing*{\section}{0pt}{0.5\baselineskip}{0.25\baselineskip}

\titleformat{\subsection}[block]{\normalfont\large\bfseries}{}{0em}{}
\titlespacing*{\subsection}{0pt}{0.5\baselineskip}{0.25\baselineskip}

% Paragraph formatting
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.5\baselineskip}

% Custom author/affil block commands (condensed)
\newcommand{\affil}[2]{\textsuperscript{#1}\textit{#2}}
\newcommand{\corresponding}[1]{\textit{*Corresponding author: #1}}

% For SI units
\RequirePackage{siunitx}

% For symbol formatting (italic)
\RequirePackage{amssymb}
\newcommand{\sym}[1]{\ensuremath{\mathit{#1}}}

\endinput