---
layout: leetcode-entry
title: "1361. Validate Binary Tree Nodes"
permalink: "/leetcode/problem/2023-10-17-1361-validate-binary-tree-nodes/"
leetcode_ui: true
entry_slug: "2023-10-17-1361-validate-binary-tree-nodes"
---

[1361. Validate Binary Tree Nodes](https://leetcode.com/problems/validate-binary-tree-nodes/description/) medium
[blog post](https://leetcode.com/problems/validate-binary-tree-nodes/solutions/4177318/kotlin-union-find/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/17102023-1361-validate-binary-tree?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/6609acfb.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/373

#### Problem TLDR

Is Binary Tree of `leftChild[]` & `rightChild[]` valid

#### Intuition

There are some examples:
![image.png](/assets/leetcode_daily_images/8ddd7dc9.webp)

Tree is valid if:
* all the leafs are connected
* there is no leaf with more than one in nodes

#### Approach

For connections check let's use Union-Find.
We also must count in nodes.

#### Complexity

- Time complexity:
$$O(an)$$, a - is for Union-Find search, it is less than 10 for Int.MAX_VALUE nodes, if we implement ranks in Union-Find

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun validateBinaryTreeNodes(n: Int, leftChild: IntArray, rightChild: IntArray): Boolean {
      val uf = IntArray(n) { it }
      val indeg = IntArray(n)
      fun root(x: Int): Int = if (x == uf[x]) x else root(uf[x])
      fun connect(a: Int, b: Int): Boolean {
        if (b < 0) return true
        if (indeg[b]++ > 0) return false
        val rootA = root(a)
        val rootB = root(b)
        uf[rootA] = rootB
        return rootA != rootB
      }
      return (0..<n).all {
        connect(it, leftChild[it]) && connect(it, rightChild[it])
      } && (0..<n).all { root(0) == root(it) }
    }

```

