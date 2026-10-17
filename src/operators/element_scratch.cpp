#include "operators/element_scratch.hpp"

using namespace mfem;

namespace incns
{

ElementScratch::ElementScratch(const ParFiniteElementSpace& vfes)
{
   const Operator* r =
      vfes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   size_ = r ? r->Height() : 0;
   for (Vector& b : buf_)
   {
      b.SetSize(size_, Device::GetDeviceMemoryType());
      b.UseDevice(true);
   }
}

void ElementScratch::Lend(int which, int n, Vector& v)
{
   MFEM_VERIFY(which == 0 || which == 1, "element_scratch: buffer 0 or 1");
   MFEM_VERIFY(n >= 0 && n <= size_, "element_scratch: " << n << " entries "
               "requested, the buffers hold " << size_);
   v.Destroy();
   v.MakeRef(buf_[which], 0, n);
   v.UseDevice(true);
}

} // namespace incns
