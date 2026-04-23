---
layout: leetcode-entry
title: "94. Binary Tree Inorder Traversal"
permalink: "/leetcode/problem/2023-12-09-94-binary-tree-inorder-traversal/"
leetcode_ui: true
entry_slug: "2023-12-09-94-binary-tree-inorder-traversal"
---

[94. Binary Tree Inorder Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/description/) easy
[blog post](https://leetcode.com/problems/binary-tree-inorder-traversal/solutions/4381000/kotlin-recursion/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/09122023-94-binary-tree-inorder-traversal?r=2bam17&utm_campaign=post&utm_medium=web)
[youtube](https://youtu.be/O2NK3P6h3QE)
![image.png](/assets/leetcode_daily_images/e58aef7c.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/433

#### Problem TLDR

Inorder traversal

#### Intuition

Nothing special. For the iterative solution we can use Morris traversal.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```koltin

  fun inorderTraversal(root: TreeNode?): List<Int> = root?.run {
    inorderTraversal(left) + listOf(`val`) + inorderTraversal(right)
  } ?: listOf<Int>()

```

