---
layout: leetcode-entry
title: "958. Check Completeness of a Binary Tree"
permalink: "/leetcode/problem/2023-03-15-958-check-completeness-of-a-binary-tree/"
leetcode_ui: true
entry_slug: "2023-03-15-958-check-completeness-of-a-binary-tree"
---

[958. Check Completeness of a Binary Tree](https://leetcode.com/problems/check-completeness-of-a-binary-tree/description/) medium

[blog post](https://leetcode.com/problems/check-completeness-of-a-binary-tree/solutions/3299207/kotlin-dfs/)

```kotlin

data class R(val min: Int, val max: Int, val complete: Boolean)
fun isCompleteTree(root: TreeNode?): Boolean {
    fun dfs(n: TreeNode): R {
        with(n) {
            if (left == null && right != null) return R(0, 0, false)
            if (left == null && right == null) return R(0, 0, true)
            val (leftMin, leftMax, leftComplete) = dfs(left)
            if (!leftComplete) return R(0, 0, false)
            if (right == null) return R(0, leftMax + 1, leftMin == leftMax && leftMin == 0)
            val (rightMin, rightMax, rightComplete) = dfs(right)
            if (!rightComplete) return R(0, 0, false)
            val isComplete = leftMin == rightMin && rightMin == rightMax
            || leftMin == leftMax && leftMin == rightMin + 1
            return R(1 + minOf(leftMin, rightMin), 1 + maxOf(leftMax, rightMax), isComplete)
        }
    }
    return root == null || dfs(root).complete
}

```

#### Join me on telegram
https://t.me/leetcode_daily_unstoppable/149
#### Intuition

![image.png](/assets/leetcode_daily_images/4bc571e7.webp)

For each node, we can compute it's left and right child `min` and `max` depth, then compare them.
#### Approach
Right depth must not be larger than left.
There are no corner cases, just be careful.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(log_2(n))$$

