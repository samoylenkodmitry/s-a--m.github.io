---
layout: leetcode-entry
title: "472. Concatenated Words"
permalink: "/leetcode/problem/2023-01-27-472-concatenated-words/"
leetcode_ui: true
entry_slug: "2023-01-27-472-concatenated-words"
---

[472. Concatenated Words](https://leetcode.com/problems/concatenated-words/description/) hard

[blog post](https://leetcode.com/problems/concatenated-words/solutions/3104496/kotlin-trie/)

```kotlin
    data class Trie(val ch: Char = '.', var isWord: Boolean = false) {
        val next = Array<Trie?>(26) { null }
        fun ind(c: Char) = c.toInt() - 'a'.toInt()
        fun exists(c: Char) = next[ind(c)] != null
        operator fun get(c: Char): Trie {
            val ind = ind(c)
            if (next[ind] == null) next[ind] = Trie(c)
            return next[ind]!!
        }
    }
    fun findAllConcatenatedWordsInADict(words: Array<String>): List<String> {
        val trie = Trie()
        words.forEach { word ->
            var t = trie
            word.forEach { t = t[it] }
            t.isWord = true
        }
        val res = mutableListOf<String>()
        words.forEach { word ->
            var tries = ArrayDeque<Pair<Trie,Int>>()
            tries.add(trie to 0)
            for (c in word) {
                repeat(tries.size) {
                    val (t, wc) = tries.poll()
                    if (t.exists(c)) {
                        val curr = t[c]
                        if (curr.isWord)  tries.add(trie to (wc + 1))
                        tries.add(curr to wc)
                    }
                }
            }
            if (tries.any { it.second > 1 && it.first === trie } ) res.add(word)
        }
        return res
    }

```

#### Telegram
https://t.me/leetcode_daily_unstoppable/99
#### Intuition
When we scan a word we must know if current suffix is a word. Trie data structure will help.

#### Approach
* first, scan all the words, and fill the Trie
* next, scan again, and for each suffix begin a new scan from the root of the trie
* preserve a word count for each of the possible suffix concatenation
#### Complexity
- Time complexity:
  $$O(nS)$$, S - is a max suffix count in one word
- Space complexity:
  $$O(n)$$

