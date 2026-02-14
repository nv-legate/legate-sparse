#include "legate_sparse/array/csr/geam.h"
#include "legate_sparse/array/csr/geam_template.inl"
#include "legate_sparse/array/csr/geam_kernels.h"

namespace sparse {
using namespace legate;

template <Type::Code INDEX_CODE, Type::Code VAL_CODE>
struct GeamComputeImplBody<VariantKind::CPU, INDEX_CODE, VAL_CODE> {
  TaskContext context;
  explicit GeamComputeImplBody(TaskContext context) : context(context) {}

  using INDEX_TY = type_of<INDEX_CODE>;
  using VAL_TY   = type_of<VAL_CODE>;

  void operator()(const AccessorRO<Rect<1>, 1>& A_pos,
                  const AccessorRO<INDEX_TY, 1>& A_crd,
                  const AccessorRO<VAL_TY, 1>& A_vals,
                  const AccessorRO<Rect<1>, 1>& B_pos,
                  const AccessorRO<INDEX_TY, 1>& B_crd,
                  const AccessorRO<VAL_TY, 1>& B_vals,
                  const AccessorRO<Rect<1>, 1>& C_pos,
                  const AccessorWO<INDEX_TY, 1>& C_crd,
                  const AccessorWO<VAL_TY, 1>& C_vals,
                  const AccessorRO<VAL_TY, 1>& alpha,
                  const AccessorRO<VAL_TY, 1>& beta,
                  const Rect<1>& rect)
  {
    VAL_TY alpha_val = alpha[0];
    VAL_TY beta_val  = beta[0];

    for (size_t row = rect.lo[0]; row < rect.hi[0] + 1; row++) {
      geam_compute_row(
        row, A_pos, A_crd, A_vals, B_pos, B_crd, B_vals, C_pos, C_crd, C_vals, alpha_val, beta_val);
    }
  }
};

template <Type::Code INDEX_CODE>
struct GeamSymbolicImplBody<VariantKind::CPU, INDEX_CODE> {
  TaskContext context;
  explicit GeamSymbolicImplBody(TaskContext context) : context(context) {}

  using INDEX_TY = type_of<INDEX_CODE>;

  void operator()(const AccessorRO<Rect<1>, 1>& A_pos,
                  const AccessorRO<INDEX_TY, 1>& A_crd,
                  const AccessorRO<Rect<1>, 1>& B_pos,
                  const AccessorRO<INDEX_TY, 1>& B_crd,
                  const AccessorRW<nnz_ty, 1>& nnz_per_row,
                  const Rect<1>& rect)
  {
    for (size_t row = rect.lo[0]; row < rect.hi[0] + 1; row++) {
      nnz_per_row[row] = geam_symbolic_row(row, A_pos, A_crd, B_pos, B_crd);
    }
  }
};

/* static */ void GeamCSRCSRSymbolic::cpu_variant(legate::TaskContext context)
{
  geam_csr_csr_symbolic_template<VariantKind::CPU>(context);
}

/* static */ void GeamCSRCSRCompute::cpu_variant(legate::TaskContext context)
{
  geam_csr_csr_compute_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static const auto sparse_reg_task_ = []() -> char {
  GeamCSRCSRSymbolic::register_variants();
  GeamCSRCSRCompute::register_variants();
  return 0;
}();

}  // namespace

}  // namespace sparse
