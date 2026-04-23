---
layout: leetcode-entry
title: "211. Design Add and Search Words Data Structure"
permalink: "/leetcode/problem/2023-03-19-211-design-add-and-search-words-data-structure/"
leetcode_ui: true
entry_slug: "2023-03-19-211-design-add-and-search-words-data-structure"
---

[211. Design Add and Search Words Data Structure](https://leetcode.com/problems/design-add-and-search-words-data-structure/description/) medium

[blog post](https://leetcode.com/problems/design-add-and-search-words-data-structure/solutions/3315405/kotlin-trie-queue/)

```kotlin

class Trie {
    val next = Array<Trie?>(26) { null }
    fun Char.ind() = toInt() - 'a'.toInt()
    operator fun get(c: Char): Trie? = next[c.ind()]
    operator fun set(c: Char, t: Trie) { next[c.ind()] = t }
    var isWord = false
}
class WordDictionary(val root: Trie = Trie()) {
    fun addWord(word: String) {
        var t = root
        word.forEach { t = t[it] ?: Trie().apply { t[it] = this } }
        t.isWord = true
    }

    fun search(word: String): Boolean = with(ArrayDeque<Trie>().apply { add(root) }) {
        !word.any { c ->
            repeat(size) {
                val t = poll()
                if (c == '.') ('a'..'z').forEach { t[it]?.let { add(it) } }
                else t[c]?.let { add(it) }
            }
            isEmpty()
        } && any { it.isWord }
    }
}

```

#### Join me on telegram
https://t.me/leetcode_daily_unstoppable/153
#### Intuition
We are already familiar with a `Trie` data structure, however there is a `wildcard` feature added. We have two options: add wildcard for every character in `addWord` method in $$O(w26^w)$$ time and then search in $$O(w)$$ time, or just add a word to `Trie` in $$O(w)$$ time and then search in $$O(w26^d)$$ time, where $$d$$ - is a wildcards count. In the description, there are at most `3` dots, so we choose the second option.

#### Approach
Let's try to write it in a Kotlin way, using as little words as possible.
#### Complexity
- Time complexity:
$$O(w)$$ add, $$O(w26^d)$$ search, where $$d$$ - wildcards count.
- Space complexity:
$$O(m)$$, $$m$$ - unique words suffixes count.

