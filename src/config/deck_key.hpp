/**
 * @file deck_key.hpp
 * @brief One entry of the deck reference: a key a YAML deck may set.
 */
#pragma once

#include <string>

namespace incns
{

/// One key a deck may set (an entry of the deck reference).
struct DeckKey
{
   std::string path;          ///< Dotted path, e.g. "time.dt".
   std::string
   type;          ///< real, int, bool, string, enum, list, real[dim], ...
   std::string
   default_value; ///< The default, as printed (from a fresh Parameters).
   std::string values;        ///< Allowed values of an enum ("a | b"); else empty.
   std::string description;   ///< What the key does.
};

} // namespace incns
