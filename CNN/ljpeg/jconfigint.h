/* libjpeg-turbo build number */
#define BUILD  "20211223"

/* Compiler's inline keyword */
#undef inline

/* How to obtain function inlining. */
#ifdef _NANO_MINGW_
#define INLINE  __inline__ __attribute__((always_inline))
#else
#define INLINE inline
#endif

/* How to obtain thread-local storage */
#ifdef _NANO_MINGW_
#define THREAD_LOCAL  __thread
#else
#define THREAD_LOCAL 
#endif

#ifdef _NANO_MINGW_
#else
#define NO_PUTENV
#define stricmp _stricmp
#endif
/* Define to the full name of this package. */
#define PACKAGE_NAME  "libjpeg-turbo"

/* Version number of package */
#define VERSION  "2.1.3"

/* The size of `size_t', as computed by sizeof. */
#define SIZEOF_SIZE_T  8

/* Define if your compiler has __builtin_ctzl() and sizeof(unsigned long) == sizeof(size_t). */
/* #undef HAVE_BUILTIN_CTZL */

/* Define to 1 if you have the <intrin.h> header file. */
/* #undef HAVE_INTRIN_H */

#if defined(_MSC_VER) && defined(HAVE_INTRIN_H)
#if (SIZEOF_SIZE_T == 8)
#define HAVE_BITSCANFORWARD64
#elif (SIZEOF_SIZE_T == 4)
#define HAVE_BITSCANFORWARD
#endif
#endif

#if defined(__has_attribute)
#if __has_attribute(fallthrough)
#define FALLTHROUGH  __attribute__((fallthrough));
#else
#define FALLTHROUGH
#endif
#else
#define FALLTHROUGH
#endif
