---
layout: leetcode-entry
title: "106. Construct Binary Tree from Inorder and Postorder Traversal"
permalink: "/leetcode/problem/2023-03-16-106-construct-binary-tree-from-inorder-and-postorder-traversal/"
leetcode_ui: true
entry_slug: "2023-03-16-106-construct-binary-tree-from-inorder-and-postorder-traversal"
---

[106. Construct Binary Tree from Inorder and Postorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/description/) medium

[blog post](https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/solutions/3303076/kotlin-dfs/)

```kotlin

fun buildTree(inorder: IntArray, postorder: IntArray): TreeNode? {
    val inToInd = inorder.asSequence().mapIndexed { i, v -> v to i }.toMap()
    var postTo = postorder.lastIndex
    fun build(inFrom: Int, inTo: Int): TreeNode? {
        if (inFrom > inTo || postTo < 0) return null
        return TreeNode(postorder[postTo]).apply {
            val inInd = inToInd[postorder[postTo]]!!
            postTo--
            right = build(inInd + 1, inTo)
            left = build(inFrom, inInd - 1)
        }
    }
    return build(0, inorder.lastIndex)
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/150
#### Intuition
Postorder traversal gives us the root of every current subtree. Next, we need to find this value in inorder traversal: from the left of it will be the left subtree, from the right - right.

#### Approach
* To more robust code, consider moving `postTo` variable as we go in the reverse-postorder: from the right to the left.
* store indices in a hashmap
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

