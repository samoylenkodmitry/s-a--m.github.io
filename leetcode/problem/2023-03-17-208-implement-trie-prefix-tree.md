---
layout: leetcode-entry
title: "208. Implement Trie (Prefix Tree)"
permalink: "/leetcode/problem/2023-03-17-208-implement-trie-prefix-tree/"
leetcode_ui: true
entry_slug: "2023-03-17-208-implement-trie-prefix-tree"
---

[208. Implement Trie (Prefix Tree)](https://leetcode.com/problems/implement-trie-prefix-tree/description/) medium

[blog post](https://leetcode.com/problems/implement-trie-prefix-tree/solutions/3306557/kotlin-just-implement-it/)

```kotlin

class Trie() {
    val root = Array<Trie?>(26) { null }
    fun Char.ind() = toInt() - 'a'.toInt()
    operator fun get(c: Char): Trie? = root[c.ind()]
    operator fun set(c: Char, v: Trie) { root[c.ind()] = v }
    var isWord = false

    fun insert(word: String) {
        var t = this
        word.forEach { t = t[it] ?: Trie().apply { t[it] = this} }
        t.isWord = true
    }

    fun String.search(): Trie? {
        var t = this@Trie
        forEach { t = t[it] ?: return@search null }
        return t
    }

    fun search(word: String) = word.search()?.isWord ?: false

    fun startsWith(prefix: String) = prefix.search() != null

}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/151
#### Intuition
Trie is a common known data structure and all must know how to implement it.

#### Approach
Let's try to write it Kotlin-way
#### Complexity
- Time complexity:
$$O(w)$$ access for each method call, where $$w$$ - is a word length
- Space complexity:
$$O(w*N)$$, where $$N$$ - is a unique words count.

