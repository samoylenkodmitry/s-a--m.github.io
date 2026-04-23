---
layout: leetcode-entry
title: "1161. Maximum Level Sum of a Binary Tree"
permalink: "/leetcode/problem/2023-06-15-1161-maximum-level-sum-of-a-binary-tree/"
leetcode_ui: true
entry_slug: "2023-06-15-1161-maximum-level-sum-of-a-binary-tree"
---

[1161. Maximum Level Sum of a Binary Tree](https://leetcode.com/problems/maximum-level-sum-of-a-binary-tree/description/) medium
[blog post](https://leetcode.com/problems/maximum-level-sum-of-a-binary-tree/solutions/3639491/kotlin-bfs/)
[substack](https://dmitriisamoilenko.substack.com/p/15062023-1161-maximum-level-sum-of?sd=pf)
![image.png](/assets/leetcode_daily_images/f726fe17.webp)

#### Join me on Telegram Leetcode_daily
https://t.me/leetcode_daily_unstoppable/246
#### Problem TLDR
Binary Tree level with max sum

#### Intuition
We can use Breadth-First Search to find a `sum` of each level.

#### Approach
Let's try to write it in a Kotlin style
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

#### Code

```kotlin

fun maxLevelSum(root: TreeNode?) = with(ArrayDeque<TreeNode>()) {
    root?.let { add(it) }
    generateSequence<Int> {
        if (isEmpty()) null else (1..size).map {
            with(poll()) {
                `val`.also {
                    left?.let { add(it) }
                    right?.let { add(it) }
                }
            }
        }.sum()
    }.withIndex().maxBy { it.value }?.index?.inc() ?: 0
}

```

