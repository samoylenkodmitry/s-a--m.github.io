---
layout: leetcode-entry
title: "894. All Possible Full Binary Trees"
permalink: "/leetcode/problem/2023-07-23-894-all-possible-full-binary-trees/"
leetcode_ui: true
entry_slug: "2023-07-23-894-all-possible-full-binary-trees"
---

[894. All Possible Full Binary Trees](https://leetcode.com/problems/all-possible-full-binary-trees/description/) medium
[blog post](https://leetcode.com/problems/all-possible-full-binary-trees/solutions/3804245/kotlin-brute-force/)
[substack](https://dmitriisamoilenko.substack.com/p/23072023-894-all-possible-full-binary?sd=pf)
![image.png](/assets/leetcode_daily_images/a0bb6a71.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/284

#### Problem TLDR

All possible Full Binary Trees with `n` nodes, each have both children

#### Intuition

First, if count of nodes is `even`, BFT is not possible.

Let's observe how the Trees are growing:

![image.png](/assets/leetcode_daily_images/1df95249.webp)

There are `n / 2` rounds of adding a new pair of nodes to each leaf of each Tree in the latest generation.

Some duplicate trees occur, so we need to calculate a `hash`.

#### Approach

Let's implement it in a BFS manner.
* to avoid collision of the `hash`, add some symbols to indicate a level `[...]`

#### Complexity

- Time complexity:
$$O(n^4 2^n)$$, n generations, queue size grows in 2^n manner, count of leafs grows by 1 each generation, so it's x + (x + 1) + .. + (x + n), giving n^2, another n for collection leafs, and another for hash and clone

- Space complexity:
$$O(n^3 2^n)$$

#### Code

```

    fun clone(curr: TreeNode): TreeNode = TreeNode(0).apply {
      curr.left?.let { left = clone(it) }
      curr.right?.let { right = clone(it) }
    }
    fun hash(curr: TreeNode): String =
      "[${curr.`val`} ${ curr.left?.let { hash(it) } } ${ curr.right?.let { hash(it) } }]"
    fun collectLeafs(curr: TreeNode): List<TreeNode> =
      if (curr.left == null && curr.right == null) listOf(curr)
      else collectLeafs(curr.left!!) + collectLeafs(curr.right!!)
    fun allPossibleFBT(n: Int): List<TreeNode?> = if (n % 2 == 0) listOf() else
      with (ArrayDeque<TreeNode>().apply { add(TreeNode(0)) }) {
        val added = HashSet<String>()
        repeat (n / 2) { rep ->
          repeat(size) {
            val root = poll()
            collectLeafs(root).forEach {
              it.left = TreeNode(0)
              it.right = TreeNode(0)
              if (added.add(hash(root))) add(clone(root))
              it.left = null
              it.right = null
            }
          }
        }
        toList()
      }

```

![image.png](/assets/leetcode_daily_images/45ffad76.webp)

effective solution. It can be described as "for every N generate every possible split of [0..i] [i+1..N]". Subtrees are also made of all possible combinations.

