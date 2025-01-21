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
    # https://www.falconprogrammer.co.uk/blog/2024/03/nix-cuda-shellhook/
      devShell = pkgs.mkShell {
        nativeBuildInputs = with pkgs; [
          python312Full
          # Python dependencies are managed via poetry
          poetry
          gcc
          cudaPackages.cudnn_9_3
          cudaPackages.cudatoolkit
          cudaPackages.cuda_cudart
          ruff
        ];
        shellHook = ''
        export CUDA_PATH=${pkgs.cudatoolkit}
        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${pkgs.stdenv.cc.cc.lib}/lib" # Use this if you have other libraries to link

        # Check for presence of /etc/nixos - if so, then add pkgs.linuxPackages.nvidia_x11 to LD_LIBRARY_PATH
        if [ -d /etc/nixos ]; then
          export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${pkgs.linuxPackages.nvidia_x11}/lib
          export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
        else
          echo "NixOS not detected - use nixGL to pass in nvidia drivers to scripts for cuda."
        fi

        export EXTRA_CCFLAGS="-I/usr/include"

        export POETRY_VIRTUALENVS_IN_PROJECT=true
        poetry install  # Automatically install dependencies inside the Poetry virtual environment
        source .venv/bin/activate
        '';
      };
    });
}