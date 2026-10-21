#include "config/expression.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <stdexcept>

namespace incns
{

namespace
{

// Instruction codes of the postfix program.
enum Code
{
   kConst, kVar, kNeg, kAdd, kSub, kMul, kDiv, kPow,
   // One-argument functions.
   kSin, kCos, kTan, kAsin, kAcos, kAtan, kSinh, kCosh, kTanh, kExp, kLog,
   kSqrt, kAbs,
   // Two-argument functions.
   kMin, kMax, kPowF, kAtan2
};

struct Function
{
   const char* name;
   int code;
   int args;
};

const Function kFunctions[] =
{
   {"sin", kSin, 1}, {"cos", kCos, 1}, {"tan", kTan, 1}, {"asin", kAsin, 1},
   {"acos", kAcos, 1}, {"atan", kAtan, 1}, {"sinh", kSinh, 1},
   {"cosh", kCosh, 1}, {"tanh", kTanh, 1}, {"exp", kExp, 1}, {"log", kLog, 1},
   {"sqrt", kSqrt, 1}, {"abs", kAbs, 1}, {"min", kMin, 2}, {"max", kMax, 2},
   {"pow", kPowF, 2}, {"atan2", kAtan2, 2}
};

const Function* FindFunction(const std::string& name)
{
   for (const Function& f : kFunctions)
   {
      if (name == f.name) { return &f; }
   }
   return nullptr;
}

int VariableIndex(const std::string& name)
{
   if (name == "x") { return 0; }
   if (name == "y") { return 1; }
   if (name == "z") { return 2; }
   if (name == "t") { return 3; }
   return -1;
}

// Recursive-descent parser emitting a postfix program.
class Parser
{
public:
   Parser(const std::string& s, const std::map<std::string, double>& params,
          std::vector<Expression::Op>& prog)
      : s_(s), params_(params), prog_(prog) {}

   void Parse()
   {
      Sum();
      Skip();
      if (pos_ != s_.size()) { Fail("unexpected '" + s_.substr(pos_, 1) + "'"); }
   }

   bool uses_time = false;

private:
   [[noreturn]] void Fail(const std::string& what) const
   {
      std::ostringstream m;
      m << "expression '" << s_ << "': " << what << " at position " << pos_ + 1;
      throw std::invalid_argument(m.str());
   }

   void Skip()
   {
      while (pos_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[pos_])))
      {
         ++pos_;
      }
   }

   bool Accept(char c)
   {
      Skip();
      if (pos_ < s_.size() && s_[pos_] == c) { ++pos_; return true; }
      return false;
   }

   void Emit(int code, double value = 0.0, int index = 0)
   {
      Expression::Op op;
      op.code = code;
      op.value = value;
      op.index = index;
      prog_.push_back(op);
   }

   // sum := product (('+' | '-') product)*
   void Sum()
   {
      Product();
      for (;;)
      {
         if (Accept('+')) { Product(); Emit(kAdd); }
         else if (Accept('-')) { Product(); Emit(kSub); }
         else { return; }
      }
   }

   // product := unary (('*' | '/') unary)*   ('**' is the power)
   void Product()
   {
      Unary();
      for (;;)
      {
         Skip();
         if (s_.compare(pos_, 2, "**") == 0) { return; }
         if (Accept('*')) { Unary(); Emit(kMul); }
         else if (Accept('/')) { Unary(); Emit(kDiv); }
         else { return; }
      }
   }

   // unary := ('-' | '+') unary | power
   void Unary()
   {
      if (Accept('-')) { Unary(); Emit(kNeg); return; }
      if (Accept('+')) { Unary(); return; }
      Power();
   }

   // power := primary (('^' | '**') unary)?   (right-associative)
   void Power()
   {
      Primary();
      Skip();
      if (s_.compare(pos_, 2, "**") == 0) { pos_ += 2; Unary(); Emit(kPow); }
      else if (Accept('^')) { Unary(); Emit(kPow); }
   }

   void Primary()
   {
      Skip();
      if (pos_ >= s_.size()) { Fail("unexpected end"); }
      const char c = s_[pos_];
      if (c == '(')
      {
         ++pos_;
         Sum();
         if (!Accept(')')) { Fail("missing ')'"); }
         return;
      }
      if (std::isdigit(static_cast<unsigned char>(c)) || c == '.')
      {
         const char* begin = s_.c_str() + pos_;
         char* end = nullptr;
         const double v = std::strtod(begin, &end);
         if (end == begin) { Fail("bad number"); }
         pos_ += static_cast<std::size_t>(end - begin);
         Emit(kConst, v);
         return;
      }
      if (std::isalpha(static_cast<unsigned char>(c)) || c == '_')
      {
         const std::size_t start = pos_;
         while (pos_ < s_.size() &&
                (std::isalnum(static_cast<unsigned char>(s_[pos_])) ||
                 s_[pos_] == '_'))
         {
            ++pos_;
         }
         const std::string name = s_.substr(start, pos_ - start);
         Skip();
         if (pos_ < s_.size() && s_[pos_] == '(')
         {
            const Function* f = FindFunction(name);
            if (!f) { pos_ = start; Fail("unknown function '" + name + "'"); }
            ++pos_;
            int n = 0;
            if (!Accept(')'))
            {
               do { Sum(); ++n; }
               while (Accept(','));
               if (!Accept(')')) { Fail("missing ')' after the arguments of " + name); }
            }
            if (n != f->args)
            {
               std::ostringstream m;
               m << name << " takes " << f->args << " argument(s), not " << n;
               Fail(m.str());
            }
            Emit(f->code);
            return;
         }
         const int var = VariableIndex(name);
         if (var >= 0)
         {
            if (var == 3) { uses_time = true; }
            Emit(kVar, 0.0, var);
            return;
         }
         const auto p = params_.find(name);
         if (p != params_.end()) { Emit(kConst, p->second); return; }
         if (name == "pi") { Emit(kConst, M_PI); return; }
         if (name == "e") { Emit(kConst, std::exp(1.0)); return; }
         pos_ = start;
         Fail("unknown name '" + name + "' (variables x, y, z, t; constants "
              "pi, e; or a parameter)");
      }
      Fail("unexpected '" + std::string(1, c) + "'");
   }

   const std::string& s_;
   const std::map<std::string, double>& params_;
   std::vector<Expression::Op>& prog_;
   std::size_t pos_ = 0;
};

} // namespace

Expression::Expression(const std::string& text,
                       const std::map<std::string, double>& parameters)
   : text_(text)
{
   for (const auto& p : parameters)
   {
      if (VariableIndex(p.first) >= 0 || FindFunction(p.first) ||
          p.first == "pi" || p.first == "e")
      {
         throw std::invalid_argument("expression parameter '" + p.first +
                                     "' shadows a variable, constant or "
                                     "function");
      }
   }
   Parser parser(text_, parameters, prog_);
   parser.Parse();
   uses_time_ = parser.uses_time;
   // The stack depth the program reaches.
   int d = 0;
   for (const Op& op : prog_)
   {
      if (op.code == kConst || op.code == kVar) { ++d; }
      else if (op.code >= kAdd && op.code <= kPow) { --d; }
      else if (op.code >= kMin) { --d; }
      depth_ = std::max(depth_, d);
   }
}

double Expression::Eval(const mfem::Vector& x, double t) const
{
   double v[4] = {0.0, 0.0, 0.0, t};
   for (int i = 0; i < std::min(x.Size(), 3); ++i) { v[i] = x(i); }
   return Eval(v);
}

double Expression::Eval(const double v[4]) const
{
   // Expressions in decks are short: a fixed stack covers them.
   constexpr int kMaxStack = 64;
   double small[kMaxStack] = {};
   std::vector<double> big;
   double* st = small;
   if (depth_ > kMaxStack) { big.resize(depth_); st = big.data(); }
   int n = 0;
   for (const Op& op : prog_)
   {
      switch (op.code)
      {
         case kConst: st[n++] = op.value; break;
         case kVar: st[n++] = v[op.index]; break;
         case kNeg: st[n - 1] = -st[n - 1]; break;
         case kAdd: --n; st[n - 1] += st[n]; break;
         case kSub: --n; st[n - 1] -= st[n]; break;
         case kMul: --n; st[n - 1] *= st[n]; break;
         case kDiv: --n; st[n - 1] /= st[n]; break;
         case kPow:
         case kPowF: --n; st[n - 1] = std::pow(st[n - 1], st[n]); break;
         case kMin: --n; st[n - 1] = std::min(st[n - 1], st[n]); break;
         case kMax: --n; st[n - 1] = std::max(st[n - 1], st[n]); break;
         case kAtan2: --n; st[n - 1] = std::atan2(st[n - 1], st[n]); break;
         case kSin: st[n - 1] = std::sin(st[n - 1]); break;
         case kCos: st[n - 1] = std::cos(st[n - 1]); break;
         case kTan: st[n - 1] = std::tan(st[n - 1]); break;
         case kAsin: st[n - 1] = std::asin(st[n - 1]); break;
         case kAcos: st[n - 1] = std::acos(st[n - 1]); break;
         case kAtan: st[n - 1] = std::atan(st[n - 1]); break;
         case kSinh: st[n - 1] = std::sinh(st[n - 1]); break;
         case kCosh: st[n - 1] = std::cosh(st[n - 1]); break;
         case kTanh: st[n - 1] = std::tanh(st[n - 1]); break;
         case kExp: st[n - 1] = std::exp(st[n - 1]); break;
         case kLog: st[n - 1] = std::log(st[n - 1]); break;
         case kSqrt: st[n - 1] = std::sqrt(st[n - 1]); break;
         case kAbs: st[n - 1] = std::abs(st[n - 1]); break;
      }
   }
   return st[0];
}

VectorExpression::VectorExpression(const std::vector<std::string>& components,
                                   const std::map<std::string, double>& parameters)
{
   c_.reserve(components.size());
   for (const std::string& s : components) { c_.emplace_back(s, parameters); }
}

void VectorExpression::Eval(const mfem::Vector& x, double t,
                            mfem::Vector& v) const
{
   v.SetSize(Size());
   for (int i = 0; i < Size(); ++i) { v(i) = c_[i].Eval(x, t); }
}

std::unique_ptr<mfem::VectorCoefficient>
VectorExpression::Coefficient(const std::shared_ptr<const VectorExpression>&
                              self)
{
   return std::make_unique<mfem::VectorFunctionCoefficient>(
             self->Size(),
             [self](const mfem::Vector & x, double t, mfem::Vector & v)
   {
      self->Eval(x, t, v);
   });
}

} // namespace incns
