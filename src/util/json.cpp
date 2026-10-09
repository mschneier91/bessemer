#include "util/json.hpp"

#include <cmath>
#include <cstdio>

namespace incns
{

Json& Json::operator[](const std::string& key)
{
   if (kind_ != Kind::Object) { kind_ = Kind::Object; members_.clear(); }
   for (auto& m : members_)
   {
      if (m.first == key) { return *m.second; }
   }
   members_.emplace_back(key, std::make_unique<Json>());
   return *members_.back().second;
}

void Json::Set(double v) { kind_ = Kind::Number; d_ = v; }
void Json::Set(long long v) { kind_ = Kind::Integer; i_ = v; }
void Json::Set(bool v) { kind_ = Kind::Bool; b_ = v; }
void Json::Set(const std::string& v) { kind_ = Kind::String; s_ = v; }

namespace
{
std::string Escape(const std::string& s)
{
   std::string o = "\"";
   for (char c : s)
   {
      switch (c)
      {
         case '"': o += "\\\""; break;
         case '\\': o += "\\\\"; break;
         case '\n': o += "\\n"; break;
         case '\t': o += "\\t"; break;
         default:
            if (static_cast<unsigned char>(c) < 0x20)
            {
               char buf[8];
               std::snprintf(buf, sizeof(buf), "\\u%04x", c);
               o += buf;
            }
            else { o += c; }
      }
   }
   return o + "\"";
}
} // namespace

void Json::DumpTo(std::string& out, int indent, int depth) const
{
   switch (kind_)
   {
      case Kind::Null: out += "null"; return;
      case Kind::Bool: out += b_ ? "true" : "false"; return;
      case Kind::Integer: out += std::to_string(i_); return;
      case Kind::String: out += Escape(s_); return;
      case Kind::Number:
      {
         if (!std::isfinite(d_)) { out += "null"; return; }
         char buf[32];
         std::snprintf(buf, sizeof(buf), "%.12g", d_);
         out += buf;
         return;
      }
      case Kind::Object:
      {
         if (members_.empty()) { out += "{}"; return; }
         const std::string pad(indent * (depth + 1), ' ');
         out += "{\n";
         for (std::size_t k = 0; k < members_.size(); ++k)
         {
            out += pad + Escape(members_[k].first) + ": ";
            members_[k].second->DumpTo(out, indent, depth + 1);
            out += (k + 1 < members_.size()) ? ",\n" : "\n";
         }
         out += std::string(indent * depth, ' ') + "}";
         return;
      }
   }
}

std::string Json::Dump(int indent) const
{
   std::string out;
   DumpTo(out, indent, 0);
   return out + "\n";
}

} // namespace incns
