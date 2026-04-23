---
layout: leetcode-entry
title: "515. Find Largest Value in Each Tree Row"
permalink: "/leetcode/problem/2023-10-24-515-find-largest-value-in-each-tree-row/"
leetcode_ui: true
entry_slug: "2023-10-24-515-find-largest-value-in-each-tree-row"
---

[515. Find Largest Value in Each Tree Row](https://leetcode.com/problems/find-largest-value-in-each-tree-row/description/) medium
[blog post](https://leetcode.com/problems/find-largest-value-in-each-tree-row/solutions/4201719/kotlin-bfs/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/24102023-515-find-largest-value-in?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/b2b7c70b.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/380

#### Problem TLDR

Binary Tree's maxes of the levels

#### Intuition

Just use Breadth-First Search

#### Approach

Let's use some Kotlin's API:
* generateSequence
* maxOf

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun largestValues(root: TreeNode?): List<Int> =
    with(ArrayDeque<TreeNode>()) {
      root?.let { add(it) }
      generateSequence { if (isEmpty()) null else
        (1..size).maxOf {
          with(removeFirst()) {
            left?.let { add(it) }
            right?.let { add(it) }
            `val`
          }
        }
      }.toList()
    }

```

