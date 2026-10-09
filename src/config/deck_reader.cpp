#include "config/deck_reader.hpp"

#include <algorithm>
#include <cstdio>
#include <map>

namespace incns
{

namespace
{
// Levenshtein distance (small strings: deck keys).
std::size_t EditDistance(const std::string& a, const std::string& b)
{
   std::vector<std::size_t> prev(b.size() + 1), cur(b.size() + 1);
   for (std::size_t j = 0; j <= b.size(); ++j) { prev[j] = j; }
   for (std::size_t i = 1; i <= a.size(); ++i)
   {
      cur[0] = i;
      for (std::size_t j = 1; j <= b.size(); ++j)
      {
         cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1,
                            prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1)});
      }
      std::swap(prev, cur);
   }
   return prev[b.size()];
}

// Split "a.b.c" into ("a.b", "c").
std::pair<std::string, std::string> SplitPath(const std::string& path)
{
   const std::size_t dot = path.rfind('.');
   if (dot == std::string::npos) { return {"", path}; }
   return {path.substr(0, dot), path.substr(dot + 1)};
}
} // namespace

DeckReader::DeckReader(YAML::Node root) : root_(std::move(root))
{
   MFEM_VERIFY(!root_ || root_.IsMap() || root_.IsNull(),
               "parameters: a deck must be a YAML map of sections");
}

DeckSection DeckReader::Root() const
{
   DeckSection s;
   s.node = root_;
   s.present = root_ && root_.IsMap();
   return s;
}

DeckSection DeckReader::Section(const DeckSection& parent, const char* name,
                                const char* description)
{
   DeckSection s;
   s.path = parent.KeyPath(name);
   s.present = parent.Has(name) && parent.Get(name).IsMap();
   if (s.present) { s.node = parent.Get(name); }
   sections_.insert(s.path);
   if (parent.path.empty()) // top-level sections head the reference's tables
   {
      DeckKey k;
      k.path = s.path;
      k.type = "section";
      k.description = description;
      keys_.push_back(k);
   }
   return s;
}

void DeckReader::Declare(const DeckSection& s, const char* key,
                         const std::string& type, const std::string& default_value,
                         const std::string& values, const char* description)
{
   Record(s, key, type, default_value, values, description);
   if (type == "list") { lists_.insert(s.KeyPath(key)); }
}

void DeckReader::Record(const DeckSection& s, const char* key,
                        const std::string& type, const std::string& default_value,
                        const std::string& values, const char* description)
{
   const std::string path = s.KeyPath(key);
   MFEM_VERIFY(key_set_.insert(path).second,
               "deck_reader: key '" << path << "' declared twice");
   keys_.push_back({path, type, default_value, values, description});
}

std::string DeckReader::ToString(double v)
{
   char buf[32];
   std::snprintf(buf, sizeof(buf), "%g", v);
   return buf;
}

std::string DeckReader::ToString(const std::vector<int>& v)
{
   std::string s = "[";
   for (std::size_t i = 0; i < v.size(); ++i)
   {
      s += (i ? ", " : "") + std::to_string(v[i]);
   }
   return s + "]";
}

std::string DeckReader::Suggest(const std::string& path) const
{
   const auto [parent, leaf] = SplitPath(path);
   std::string best;
   std::size_t best_d = std::max<std::size_t>(2, leaf.size() / 3) + 1;
   auto consider = [&](const std::string & cand)
   {
      const auto [cp, cl] = SplitPath(cand);
      if (cp != parent) { return; }
      const std::size_t d = EditDistance(leaf, cl);
      if (d < best_d) { best_d = d; best = cand; }
   };
   for (const std::string& k : key_set_) { consider(k); }
   for (const std::string& k : sections_) { consider(k); }
   return best;
}

void DeckReader::Walk(const YAML::Node& node, const std::string& path,
                      std::vector<std::string>& out) const
{
   for (const auto& kv : node)
   {
      const std::string key = kv.first.as<std::string>();
      const std::string p = path.empty() ? key : path + "." + key;
      if (lists_.count(p)) { continue; } // checked by its own parser
      const bool is_section = sections_.count(p) > 0;
      if (!key_set_.count(p) && !is_section)
      {
         const std::string s = Suggest(p);
         out.push_back(p + (s.empty() ? "" : " (did you mean " + s + "?)"));
         continue;
      }
      if (is_section && kv.second.IsMap()) { Walk(kv.second, p, out); }
   }
}

std::vector<std::string> DeckReader::UnknownKeys() const
{
   std::vector<std::string> out;
   if (root_ && root_.IsMap()) { Walk(root_, "", out); }
   return out;
}

std::string DeckReader::Markdown(const std::vector<DeckKey>& keys)
{
   // Group by top-level section, in declaration order.
   std::vector<std::string> order;
   std::map<std::string, std::vector<const DeckKey*>> groups;
   std::map<std::string, std::string> section_desc;
   for (const DeckKey& k : keys)
   {
      const std::size_t dot = k.path.find('.');
      std::string top = (dot == std::string::npos) ? "" : k.path.substr(0, dot);
      if (k.type == "section" && dot == std::string::npos)
      {
         top = k.path;
         section_desc[top] = k.description;
      }
      if (!groups.count(top)) { order.push_back(top); }
      auto& g = groups[top];
      if (!(k.type == "section" && dot == std::string::npos)) { g.push_back(&k); }
   }
   auto cell = [](std::string s)
   {
      std::string o;
      for (char c : s) { o += (c == '|') ? std::string("\\|") : std::string(1, c); }
      return o;
   };
   std::string md =
      "# Deck reference\n\n"
      "Every key a YAML deck (`apps/run_case <deck.yaml>`) understands, with its "
      "default. Omitted keys keep their defaults; an unknown key is an error that "
      "names the closest valid key.\n\n"
      "**Generated** by `build/cpu/apps/run_case --deck-reference` from the loader "
      "itself (`src/config/parameters.cpp`); `deck_test` fails when this file is "
      "out of date. Do not edit by hand.\n";
   for (const std::string& top : order)
   {
      md += "\n## " + (top.empty() ? std::string("Top level") : "`" + top + "`") +
            "\n\n";
      if (section_desc.count(top)) { md += section_desc[top] + "\n\n"; }
      md += "| Key | Type | Default | Description |\n|---|---|---|---|\n";
      for (const DeckKey* k : groups[top])
      {
         std::string type = k->type;
         if (!k->values.empty()) { type = k->values; }
         md += "| `" + k->path + "` | " + cell(type) + " | " +
               (k->type == "section" ? std::string("") : "`" + cell(k->default_value) + "`") +
               " | " + cell(k->description) + " |\n";
      }
   }
   return md;
}

} // namespace incns
