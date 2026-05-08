#ifndef attr_reader_val
#define attr_reader_val(_var, _func) \
[[nodiscard]] auto _func() const { return _var; }
#endif

#ifndef attr_writer_val
#define attr_writer_val(_var, _func) \
template <typename T>              \
void _func(const T& value) {       \
_var = value;                    \
}
#endif

#ifndef attr_reader_ref
#define attr_reader_ref(_var, _func) \
[[nodiscard]] auto&& _func() const { return _var; }
#endif

#ifndef REF_IN
#define REF_IN const&
#endif

#ifndef PTR_IN
#define PTR_IN const*
#endif

#ifndef FWD_IN
#define FWD_IN &&
#endif

#ifndef REF_OUT
#define REF_OUT &
#endif

#ifndef PTR_OUT
#define PTR_OUT *
#endif