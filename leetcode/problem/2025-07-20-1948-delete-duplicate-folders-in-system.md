---
layout: leetcode-entry
title: "1948. Delete Duplicate Folders in System"
permalink: "/leetcode/problem/2025-07-20-1948-delete-duplicate-folders-in-system/"
leetcode_ui: true
entry_slug: "2025-07-20-1948-delete-duplicate-folders-in-system"
---

[1948. Delete Duplicate Folders in System](https://leetcode.com/problems/delete-duplicate-folders-in-system/description/) hard
[blog post](https://leetcode.com/problems/delete-duplicate-folders-in-system/solutions/6981453/kotlin-by-samoylenkodmitry-jgq4/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/20072025-1948-delete-duplicate-folders?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/MeraNuFL7oo)
![1.webp](/assets/leetcode_daily_images/390887af.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1055

#### Problem TLDR

Remove duplicate folder trees #hard #trie

#### Intuition

```j
    // a
    // c
    // d
    // a/b
    // c/b
    // d/a
    // go backwards: corner case is collision `a with d/a`
    // first avoid parents: a vs a/b, we should skip a, for same startswith, pick longest
    // too complex, maybe should go forward with trie

```
The hardness is the correct implementation:
* build the tree keys
* count them
* remove all with count > 1

#### Approach

* we can do two DFS or a single DFS but with worse time complexity to merge indices

#### Complexity

- Time complexity:
$$O(nlogn)$$, worst case is n^2 for long graph nodes

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 189ms
    fun deleteDuplicateFolder(p: List<List<String>>) = buildList<List<String>> {
        class T(val i: HashSet<Int> = HashSet()): TreeMap<String, T>()
        val r = T(); val m = HashMap<String, T>()
        for (j in p.indices) p[j].fold(r) { t, f -> t.getOrPut(f, ::T).also { it.i += j }}
        fun dfs(t: T): String = t.map { (f, nt) -> "$f[${ dfs(nt) }]" }.toString().also { k ->
            if (t !== r) for (nt in t.values) t.i += nt.i
            if (t.size > 0 && k in m) r.i += m[k]!!.i + t.i else m[k] = t
        }
        dfs(r); for (i in p.indices) if (i !in r.i) add(p[i])
    }

```

