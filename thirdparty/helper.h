#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

const char* get_file_name(const char* file_path);
char* shift_args(int* argc, char*** argv);
void check(bool expr, const char* errormsg, ...);

#ifdef HELPER_IMPLEMENTATION
const char* get_file_name(const char* file_path)
{
    // windows
    const char* res = strrchr(file_path, '\\');
    if (!res) {
        // linux
        res = strrchr(file_path, '/');
    }

    return res ? res+1 : file_path;
}

char* shift_args(int* argc, char*** argv)
{
    char* result = *argv[0];
    *argc -= 1;
    *argv += 1;
    return result; 
}

void check(bool expr, const char* errormsg, ...)
{
    if (expr) {
        va_list args;
        va_start(args, errormsg);
        fprintf(stderr, "ERROR: ");
        vfprintf(stderr, errormsg, args);
        va_end(args);
        fprintf(stderr, "\n");
        exit(EXIT_FAILURE);
    }
}
#endif
