---
layout: leetcode-entry
title: "652. Find Duplicate Subtrees"
permalink: "/leetcode/problem/2023-02-28-652-find-duplicate-subtrees/"
leetcode_ui: true
entry_slug: "2023-02-28-652-find-duplicate-subtrees"
---

[652. Find Duplicate Subtrees](https://leetcode.com/problems/find-duplicate-subtrees/description/) medium

[blog post](https://leetcode.com/problems/find-duplicate-subtrees/solutions/3239077/kotlin-preorder-hashset/)

```kotlin

fun findDuplicateSubtrees(root: TreeNode?): List<TreeNode?> {
    val result = mutableListOf<TreeNode?>()
    val hashes = HashSet<String>()
        val added = HashSet<String>()
            fun hashDFS(node: TreeNode): String {
                return with(node) {
                    "[" + (left?.let { hashDFS(it) } ?: "*") +
                    "_" + `val` + "_" +
                    (right?.let { hashDFS(it) } ?: "*") + "]"
                }.also {
                    if (!hashes.add(it) && added.add(it)) result.add(node)
                }
            }
            if (root != null) hashDFS(root)
            return result
        }

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/132
#### Intuition
We can traverse the tree and construct a hash of each node, then just compare nodes with equal hashes. Another way is to serialize the tree and compare that data.

#### Approach
Let's use pre-order traversal and serialize each node into string, also add that into `HashSet` and check for duplicates.
#### Complexity
- Time complexity:
$$O(n^2)$$, because of the string construction on each node.
- Space complexity:
$$O(n^2)$$

