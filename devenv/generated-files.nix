{ ... }:

{
  files.".devenv/generated/compiler-probes/valid.c".text = ''
    int main(void) {
        return 0;
    }
  '';

  files.".devenv/generated/compiler-probes/warning.c".text = ''
    int main(void) {
        int unused = 0;
        return 0;
    }
  '';

  files.".devenv/generated/compiler-probes/valid.cpp".text = ''
    int main() {
        return 0;
    }
  '';
}
