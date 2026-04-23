---
layout: leetcode-entry
title: "839. Similar String Groups"
permalink: "/leetcode/problem/2023-04-28-839-similar-string-groups/"
leetcode_ui: true
entry_slug: "2023-04-28-839-similar-string-groups"
---

[839. Similar String Groups](https://leetcode.com/problems/similar-string-groups/description/) hard

```kotlin

fun numSimilarGroups(strs: Array<String>): Int {
    fun similar(i: Int, j: Int): Boolean {
        var from = 0
        while (from < strs[i].length && strs[i][from] == strs[j][from]) from++
        var to = strs[i].lastIndex
        while (to >= 0 && strs[i][to] == strs[j][to]) to--
        for (x in from + 1..to - 1)
        if (strs[i][x] != strs[j][x]) return false
        return true
    }
    val uf = IntArray(strs.size) { it }
    fun root(x: Int): Int {
        var n = x
        while (uf[n] != n) n = uf[n]
        uf[x] = n
        return n
    }
    var groups = strs.size
    fun union(a: Int, b: Int) {
        val rootA = root(a)
        val rootB = root(b)
        if (rootA != rootB) {
            groups--
            uf[rootB] = rootA
        }
    }
    for (i in 0..strs.lastIndex)
    for (j in i + 1..strs.lastIndex)
    if (similar(i, j)) union(i, j)
    return groups
}

```

[blog post](https://leetcode.com/problems/similar-string-groups/solutions/3462309/kotlin-union-find/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-28042023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/194
#### Intuition
For tracking the groups, Union-Find is a good start. Next, we need to compare the similarity of each to each word, that is $$O(n^2)$$.
For the similarity, we need a linear algorithm. Let's divide the words into three parts: `prefix+a+body+b+suffix`. Two words are similar if their `prefix`, `suffix` and `body` are similar, leaving the only different letters `a` and `b`.

#### Approach
* decrease the groups when the two groups are joined together
* shorten the Union-Find root's path `uf[x] = n`
* more complex Union-Find algorithm with `ranks` give the optimal time of $$O(lg*n)$$, where `lg*n` is the inverse Ackerman function. It is inverse of the f(n) = 2^2^2^2..n times.
#### Complexity
- Time complexity:
$$O(n^2a(n))$$
- Space complexity:
$$O(n)$$

