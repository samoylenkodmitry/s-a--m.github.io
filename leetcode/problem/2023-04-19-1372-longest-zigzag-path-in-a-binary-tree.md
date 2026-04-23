---
layout: leetcode-entry
title: "1372. Longest ZigZag Path in a Binary Tree"
permalink: "/leetcode/problem/2023-04-19-1372-longest-zigzag-path-in-a-binary-tree/"
leetcode_ui: true
entry_slug: "2023-04-19-1372-longest-zigzag-path-in-a-binary-tree"
---

[1372. Longest ZigZag Path in a Binary Tree](https://leetcode.com/problems/longest-zigzag-path-in-a-binary-tree/description/) medium

```kotlin

fun longestZigZag(root: TreeNode?): Int {
    var max = 0
    fun dfs(n: TreeNode?, len: Int, dir: Int) {
        max = maxOf(max, len)
        if (n == null) return@dfs
        when (dir) {
            0 -> {
                dfs(n?.left, 0, -1)
                dfs(n?.right, 0, 1)
            }
            1 -> {
                dfs(n?.left, len + 1, -1)
                dfs(n?.right, 0, 1)
            }
            -1 -> {
                dfs(n?.right, len + 1, 1)
                dfs(n?.left, 0, -1)
            }
        }
    }
    dfs(root, 0, 0)
    return max
}

```

[blog post](https://leetcode.com/problems/longest-zigzag-path-in-a-binary-tree/solutions/3433418/kotlin-dfs/?orderBy=most_votes)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-19042023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/185
#### Intuition
Search all the possibilities with DFS

#### Approach
Compute the `max` as you go
#### Complexity
- Time complexity:
$$O(nlog_2(n))$$, for each level of `height` we traverse the full tree
- Space complexity:
$$O(log_2(n))$$

