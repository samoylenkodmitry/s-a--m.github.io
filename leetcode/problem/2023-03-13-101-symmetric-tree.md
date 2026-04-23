---
layout: leetcode-entry
title: "101. Symmetric Tree"
permalink: "/leetcode/problem/2023-03-13-101-symmetric-tree/"
leetcode_ui: true
entry_slug: "2023-03-13-101-symmetric-tree"
---

[101. Symmetric Tree](https://leetcode.com/problems/symmetric-tree/description/) easy

[blog post](https://leetcode.com/problems/symmetric-tree/solutions/3291127/kotlin-bfs-recursion/)

```kotlin

data class H(val x: Int?)
fun isSymmetric(root: TreeNode?): Boolean {
    with(ArrayDeque<TreeNode>().apply { root?.let { add(it) } }) {
        while (isNotEmpty()) {
            val stack = Stack<H>()
                val sz = size
                repeat(sz) {
                    if (sz == 1 && peek().left?.`val` != peek().right?.`val`) return false
                    with(poll()) {
                        if (sz == 1 || it < sz / 2) {
                            stack.push(H(left?.`val`))
                            stack.push(H(right?.`val`))
                        } else {
                            if (stack.isEmpty() || stack.pop().x != left?.`val`) return false
                            if (stack.isEmpty() || stack.pop().x != right?.`val`) return false
                        }
                        left?.let { add(it)}
                        right?.let { add(it)}
                    }
                }
            }
        }
        return true
    }

    fun isSymmetric2(root: TreeNode?): Boolean {
        fun isSymmetric(leftRoot: TreeNode?, rightRoot: TreeNode?): Boolean {
            return leftRoot == null && rightRoot == null
            || leftRoot != null && rightRoot != null
            && leftRoot.`val` == rightRoot.`val`
            && isSymmetric(leftRoot.left, rightRoot.right)
            && isSymmetric(leftRoot.right, rightRoot.left)
        }
        return isSymmetric(root?.left, root?.right)
    }

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/147
#### Intuition
Recursive solution based on idea that we must compare `left.left` with `right.right` and `left.right` with `right.left`.
Iterative solution is just BFS and Stack.

#### Approach
Recursive: just write helper function.
Iterative: save also `null`'s to solve corner cases.
#### Complexity
- Time complexity:
Recursive: $$O(n)$$
Iterative: $$O(n)$$
- Space complexity:
Recursive: $$O(log_2(n))$$
Iterative: $$O(n)$$

