{
  description = "Python project with PyTorch";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
    in {
      devShell = pkgs.mkShell {
        buildInputs = with pkgs; [
          python312Full
          # Python dependencies are managed via poetry
          poetry
          gcc
          cudaPackages.cudnn
          cudaPackages.cudatoolkit
          linuxPackages.nvidia_x11
          ruff
        ];
        shellHook = ''
        export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
        export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
        export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
        export EXTRA_CCFLAGS="-I/usr/include"

        export POETRY_VIRTUALENVS_IN_PROJECT=true
        poetry install  # Automatically install dependencies inside the Poetry virtual environment
        source .venv/bin/activate
        '';
      };
    });
}