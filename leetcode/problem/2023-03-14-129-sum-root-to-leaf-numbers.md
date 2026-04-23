---
layout: leetcode-entry
title: "129. Sum Root to Leaf Numbers"
permalink: "/leetcode/problem/2023-03-14-129-sum-root-to-leaf-numbers/"
leetcode_ui: true
entry_slug: "2023-03-14-129-sum-root-to-leaf-numbers"
---

[129. Sum Root to Leaf Numbers](https://leetcode.com/problems/sum-root-to-leaf-numbers/description/) medium

[blog post](https://leetcode.com/problems/sum-root-to-leaf-numbers/solutions/3295054/kotlin-dfs/)

```kotlin

fun sumNumbers(root: TreeNode?): Int = if (root == null) 0 else {
    var sum = 0
    fun dfs(n: TreeNode, soFar: Int) {
        with(n) {
            val x = soFar * 10 + `val`
            if (left == null && right == null) sum += x
            if (left != null) dfs(left, x)
            if (right != null) dfs(right, x)
        }
    }
    dfs(root, 0)

    sum
}

```

#### Join me on telegram
https://t.me/leetcode_daily_unstoppable/148
#### Intuition
Just make DFS and add to the sum if the node is a leaf.

#### Approach
The most trivial way is to keep `sum` variable outside the dfs function.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(log_2(n))$$

