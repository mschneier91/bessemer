/**
 * @file deck_reader.hpp
 * @brief Reads a YAML deck key by key while recording the schema: every key's
 *        path, type, default, allowed values and description. Running the
 *        loader on an empty deck yields the complete schema (the deck
 *        reference), and keys a deck sets that no one declared are reported.
 */
#pragma once

#include "config/deck_key.hpp"
#include "mfem.hpp" // MFEM_VERIFY / MFEM_ABORT

#include <yaml-cpp/yaml.h>

#include <array>
#include <initializer_list>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace incns
{

/// A deck section: its YAML node (when the deck has it) and its dotted path.
struct DeckSection
{
   YAML::Node node;     ///< The section's node (meaningful only when present).
   bool present = false; ///< Whether the deck has this section.
   std::string path;    ///< Dotted path ("" for the top level).

   /**
    * @param key Key in this section.
    * @return Whether the deck sets @p key here.
    */
   bool Has(const char* key) const { return present && node[key]; }
   /**
    * @param key Key in this section. @pre Has(key).
    * @return Its node.
    */
   YAML::Node Get(const char* key) const { return node[key]; }
   /**
    * @param key Key in this section.
    * @return Its dotted path.
    */
   std::string KeyPath(const char* key) const
   {
      return path.empty() ? std::string(key) : path + "." + key;
   }
};

/**
 * @brief Reads deck values into fields and records each key's schema entry.
 *
 * Every key is declared whether or not the deck sets it, so one pass over an
 * empty deck lists the whole schema with the defaults of a fresh Parameters.
 */
class DeckReader
{
public:
   /// @param root The deck (a map; an empty map yields the schema only).
   explicit DeckReader(YAML::Node root);

   /// @return The top-level section.
   DeckSection Root() const;

   /**
    * @brief Declare a sub-section (a YAML map).
    * @param parent      Enclosing section.
    * @param name        Section key.
    * @param description What the section configures.
    * @return The section (present only if the deck has it).
    */
   DeckSection Section(const DeckSection& parent, const char* name,
                       const char* description);

   /**
    * @brief Declare a scalar key and read it into @p value when set.
    * @param s           Section.
    * @param key         Key.
    * @param value       Field: holds the default on entry.
    * @param description What the key does.
    */
   template <typename T>
   void Read(const DeckSection& s, const char* key, T& value,
             const char* description)
   {
      Record(s, key, TypeName(value), ToString(value), "", description);
      if (s.Has(key)) { value = s.Get(key).template as<T>(); }
   }

   /**
    * @brief Declare a per-dimension array key and read its first @p count
    *        entries when set.
    * @param s           Section.
    * @param key         Key.
    * @param value       Field: holds the default on entry.
    * @param count       Entries the deck must give (the dimension).
    * @param description What the key does.
    */
   template <typename T, std::size_t N>
   void ReadArray(const DeckSection& s, const char* key, std::array<T, N>& value,
                  int count, const char* description)
   {
      std::ostringstream d;
      d << "[";
      for (int i = 0; i < count; ++i) { d << (i ? ", " : "") << ToString(value[i]); }
      d << "]";
      Record(s, key, TypeName(value[0]) + "[dim]", d.str(), "", description);
      if (!s.Has(key)) { return; }
      const YAML::Node seq = s.Get(key);
      MFEM_VERIFY(seq.IsSequence() && static_cast<int>(seq.size()) == count,
                  "parameters: '" << s.KeyPath(key) << "' must be a sequence of "
                  << count);
      for (int i = 0; i < count; ++i) { value[i] = seq[i].template as<T>(); }
   }

   /**
    * @brief Declare an enumerated key and read it when set.
    * @param s           Section.
    * @param key         Key.
    * @param value       Field: holds the default on entry.
    * @param names       Accepted spellings; the first name of each value is
    *                    the canonical one (later ones are accepted aliases).
    * @param description What the key does.
    */
   template <typename E>
   void ReadEnum(const DeckSection& s, const char* key, E& value,
                 std::initializer_list<std::pair<const char*, E>> names,
                 const char* description)
   {
      std::string canonical, def;
      std::vector<E> seen;
      for (const auto& n : names)
      {
         bool dup = false;
         for (const E& v : seen) { dup = dup || v == n.second; }
         if (dup) { continue; }
         seen.push_back(n.second);
         canonical += (canonical.empty() ? "" : " | ") + std::string(n.first);
         if (n.second == value && def.empty()) { def = n.first; }
      }
      Record(s, key, "enum", def, canonical, description);
      if (!s.Has(key)) { return; }
      const std::string got = s.Get(key).template as<std::string>();
      for (const auto& n : names)
      {
         if (got == n.first) { value = n.second; return; }
      }
      MFEM_ABORT("parameters: unknown " << s.KeyPath(key) << " '" << got
                 << "' (" << canonical << ")");
   }

   /**
    * @brief Declare a key the caller parses itself.
    * @param s             Section.
    * @param key           Key.
    * @param type          Type as printed.
    * @param default_value Default as printed.
    * @param values        Allowed values ("" if not an enum).
    * @param description   What the key does.
    */
   void Declare(const DeckSection& s, const char* key, const std::string& type,
                const std::string& default_value, const std::string& values,
                const char* description);

   /**
    * @return The keys the deck sets that were never declared, each as
    *         "path (did you mean ...?)" when a declared key is close.
    *         Entries of list-valued keys (boundary_conditions) are checked by
    *         their parser, not here.
    */
   std::vector<std::string> UnknownKeys() const;

   /// @return Every declared key, in declaration order.
   const std::vector<DeckKey>& Keys() const { return keys_; }

   /**
    * @brief The deck reference: one Markdown table per section.
    * @param keys Schema (Keys() of a reader run on an empty deck).
    * @return Markdown text.
    */
   static std::string Markdown(const std::vector<DeckKey>& keys);

   /// @param v A value of the type. @return Its type name in the reference.
   static std::string TypeName(double v) { (void)v; return "real"; }
   /// @param v A value of the type. @return Its type name in the reference.
   static std::string TypeName(int v) { (void)v; return "int"; }
   /// @param v A value of the type. @return Its type name in the reference.
   static std::string TypeName(long long v) { (void)v; return "int"; }
   /// @param v A value of the type. @return Its type name in the reference.
   static std::string TypeName(bool v) { (void)v; return "bool"; }
   /// @param v A value of the type. @return Its type name in the reference.
   static std::string TypeName(const std::string& v) { (void)v; return "string"; }
   /// @param v A value of the type. @return Its type name in the reference.
   static std::string TypeName(const std::vector<int>& v) { (void)v; return "int list"; }
   /// @param v A value. @return It, as printed in the reference.
   static std::string ToString(double v);
   /// @param v A value. @return It, as printed in the reference.
   static std::string ToString(int v) { return std::to_string(v); }
   /// @param v A value. @return It, as printed in the reference.
   static std::string ToString(long long v) { return std::to_string(v); }
   /// @param v A value. @return It, as printed in the reference.
   static std::string ToString(bool v) { return v ? "true" : "false"; }
   /// @param v A value. @return It, as printed in the reference.
   static std::string ToString(const std::string& v) { return v.empty() ? "\"\"" : v; }
   /// @param v A value. @return It, as printed in the reference.
   static std::string ToString(const std::vector<int>& v);

private:
   /**
    * @brief Record a key (aborts on a duplicate declaration).
    * @param s             Section.
    * @param key           Key.
    * @param type          Type as printed.
    * @param default_value Default as printed.
    * @param values        Allowed values ("" if not an enum).
    * @param description   What the key does.
    */
   void Record(const DeckSection& s, const char* key, const std::string& type,
               const std::string& default_value, const std::string& values,
               const char* description);
   /**
    * @brief Append the unknown keys of a map to @p out.
    * @param node A map of the deck.
    * @param path Its dotted path.
    * @param out  Unknown keys found.
    */
   void Walk(const YAML::Node& node, const std::string& path,
             std::vector<std::string>& out) const;
   /**
    * @param path An unknown key's dotted path.
    * @return The declared key at the same level closest to it (empty when
    *         none is close).
    */
   std::string Suggest(const std::string& path) const;

   YAML::Node root_;                ///< The deck.
   std::vector<DeckKey> keys_;      ///< Declared keys, in order.
   std::set<std::string> key_set_;  ///< Paths of declared keys.
   std::set<std::string> sections_; ///< Paths of declared sections.
   std::set<std::string>
   lists_;    ///< Paths of list/map-valued keys (not walked).
};

} // namespace incns
