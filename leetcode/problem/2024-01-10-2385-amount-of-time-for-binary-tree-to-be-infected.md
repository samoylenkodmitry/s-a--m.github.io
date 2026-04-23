---
layout: leetcode-entry
title: "2385. Amount of Time for Binary Tree to Be Infected"
permalink: "/leetcode/problem/2024-01-10-2385-amount-of-time-for-binary-tree-to-be-infected/"
leetcode_ui: true
entry_slug: "2024-01-10-2385-amount-of-time-for-binary-tree-to-be-infected"
---

[2385. Amount of Time for Binary Tree to Be Infected](https://leetcode.com/problems/amount-of-time-for-binary-tree-to-be-infected/description/) medium
[blog post](https://leetcode.com/problems/amount-of-time-for-binary-tree-to-be-infected/solutions/4539119/kotlin-bfs/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/10012024-2385-amount-of-time-for?r=2bam17&utm_campaign=post&utm_medium=web&showWelcome=true)
[youtube](https://youtu.be/5Ha9J4svCKc)
![image.png](/assets/leetcode_daily_images/d8ab3e28.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/467

#### Problem TLDR

Max distance from node in a Binary Tree.

#### Intuition

Let's build a graph, then do a Breadth-First Search from starting node.

#### Approach

We can store it in a `parent[TreeNode]` map or just in two directional `node to list<node>` graph.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

  fun amountOfTime(root: TreeNode?, start: Int): Int {
    val fromTo = mutableMapOf<TreeNode, MutableList<TreeNode>>()
    var queue = ArrayDeque<TreeNode>()
    val visited = mutableSetOf<TreeNode>()
    fun dfs(n: TreeNode): Unit = with (n) {
      if (`val` == start) {
        queue.add(n)
        visited.add(n)
      }
      left?.let {
        fromTo.getOrPut(n) { mutableListOf() } += it
        fromTo.getOrPut(it) { mutableListOf() } += n
        dfs(it)
      }
      right?.let {
        fromTo.getOrPut(n) { mutableListOf() } += it
        fromTo.getOrPut(it) { mutableListOf() } += n
        dfs(it)
      }
    }
    root?.let { dfs(it) }
    var time = -1
    while (queue.isNotEmpty()) {
      repeat(queue.size) {
        var x = queue.removeFirst()
        fromTo[x]?.onEach {
          if (visited.add(it)) queue.add(it)
        }
      }
      time++
    }
    return time
  }

```

