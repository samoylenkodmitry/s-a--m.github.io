---
layout: leetcode-entry
title: "144. Binary Tree Preorder Traversal"
permalink: "/leetcode/problem/2023-01-09-144-binary-tree-preorder-traversal/"
leetcode_ui: true
entry_slug: "2023-01-09-144-binary-tree-preorder-traversal"
---

[144. Binary Tree Preorder Traversal](https://leetcode.com/problems/binary-tree-preorder-traversal/description/) easy

[https://t.me/leetcode_daily_unstoppable/80](https://t.me/leetcode_daily_unstoppable/80)

[blog post](https://leetcode.com/problems/binary-tree-preorder-traversal/solutions/3023310/kotlin-morris-stack-recursive/)

```kotlin
class Solution {
    fun preorderTraversal(root: TreeNode?): List<Int> {
        val res = mutableListOf<Int>()
        var node = root
        while(node != null) {
            res.add(node.`val`)
            if (node.left != null) {
                if (node.right != null) {
                    var rightmost = node.left!!
                    while (rightmost.right != null) rightmost = rightmost.right
                    rightmost.right = node.right
                }
                node = node.left
            } else if (node.right != null) node = node.right
            else node = null
        }
        return res
    }
    fun preorderTraversalStack(root: TreeNode?): List<Int> {
        val res = mutableListOf<Int>()
        var node = root
        val rightStack = ArrayDeque<TreeNode>()
        while(node != null) {
            res.add(node.`val`)
            if (node.left != null) {
                if (node.right != null) {
                    rightStack.addLast(node.right!!) // <-- this step can be replaced with Morris
                    // traversal.
                }
                node = node.left
            } else if (node.right != null) node = node.right
            else if (rightStack.isNotEmpty()) node = rightStack.removeLast()
            else node = null
        }
        return res
    }
    fun preorderTraversalRec(root: TreeNode?): List<Int> = mutableListOf<Int>().apply {
        root?.let {
            add(it.`val`)
            addAll(preorderTraversal(it.left))
            addAll(preorderTraversal(it.right))
        }
    }

}

```

Recursive solution is a trivial. For stack solution, we need to remember each `right` node. Morris' solution use the tree modification to save each `right` node in the rightmost end of the left subtree.
Let's implement them all.

Space: O(logN) for stack, O(1) for Morris', Time: O(n)

