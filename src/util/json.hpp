/**
 * @file json.hpp
 * @brief A minimal ordered JSON tree for machine-readable run summaries:
 *        objects keep insertion order, numbers print with full precision,
 *        non-finite numbers become null.
 */
#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace incns
{

/// A JSON value: null, bool, number, string or object (ordered keys).
class Json
{
public:
   /// A null value.
   Json() = default;

   /**
    * @brief The member @p key of this object (created as null if absent);
    *        turns a null value into an object.
    * @param key Member name.
    * @return The member.
    */
   Json& operator[](const std::string& key);

   /// Set a number (non-finite values print as null). @param v Value.
   void Set(double v);
   /// Set an integer. @param v Value.
   void Set(long long v);
   /// Set an integer. @param v Value.
   void Set(int v) { Set(static_cast<long long>(v)); }
   /// Set a boolean. @param v Value.
   void Set(bool v);
   /// Set a string. @param v Value.
   void Set(const std::string& v);
   /// Set a string. @param v Value.
   void Set(const char* v) { Set(std::string(v)); }

   /**
    * @brief Serialize.
    * @param indent Spaces per nesting level.
    * @return The JSON text (with a trailing newline at the top level).
    */
   std::string Dump(int indent = 2) const;

private:
   /// The kind of value held.
   enum class Kind { Null, Bool, Number, Integer, String, Object };
   /**
    * @brief Serialize into @p out.
    * @param out    Text appended to.
    * @param indent Spaces per nesting level.
    * @param depth  Current nesting depth.
    */
   void DumpTo(std::string& out, int indent, int depth) const;

   Kind kind_ = Kind::Null; ///< What this value is.
   bool b_ = false;         ///< Bool value.
   double d_ = 0.0;         ///< Number value.
   long long i_ = 0;        ///< Integer value.
   std::string s_;          ///< String value.
   /// Object members, in insertion order.
   std::vector<std::pair<std::string, std::unique_ptr<Json>>> members_;
};

} // namespace incns
