{ pkgs, lib, config, inputs, ... }:
{
    languages = {
        python = {
            enable = true;
            package = pkgs.python313;

            uv = {
                enable = true;
                sync.enable = false;
            };
        };
        typst.enable = true;
    };


    # manylinux wheels (numpy, scipy, ...) dlopen system libs that aren't on
    # nix's loader path by default.
    env.LD_LIBRARY_PATH = "${lib.makeLibraryPath [ pkgs.zlib pkgs.stdenv.cc.cc.lib ]}:$LD_LIBRARY_PATH";
}
