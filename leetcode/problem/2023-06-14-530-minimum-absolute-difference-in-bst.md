---
layout: leetcode-entry
title: "530. Minimum Absolute Difference in BST"
permalink: "/leetcode/problem/2023-06-14-530-minimum-absolute-difference-in-bst/"
leetcode_ui: true
entry_slug: "2023-06-14-530-minimum-absolute-difference-in-bst"
---

[530. Minimum Absolute Difference in BST](https://leetcode.com/problems/minimum-absolute-difference-in-bst/description/) easy
[blog post](https://leetcode.com/problems/minimum-absolute-difference-in-bst/solutions/3635561/kotlin-morris-traversal/)
[substack](https://dmitriisamoilenko.substack.com/p/14062023-530-minimum-absolute-difference?sd=pf)
![image.png](/assets/leetcode_daily_images/52ba0d0f.webp)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/245
#### Problem TLDR
Min difference in a BST
#### Intuition
In-order traversal in a BST gives a sorted order, we can compare `curr - prev`.

#### Approach
Let's write a [Morris traversal](https://en.wikipedia.org/wiki/Threaded_binary_tree): make the current node a rightmost child of its left child.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(1)$$

#### Code

```kotlin

fun getMinimumDifference(root: TreeNode?): Int {
    if (root == null) return 0
    var minDiff = Int.MAX_VALUE
    var curr = root
    var prev = -1
    while (curr !=  null) {
        val left = curr.left
        if (left != null) {
            var leftRight = left
            while (leftRight.right != null) leftRight = leftRight.right
            leftRight.right = curr
            curr.left = null
            curr = left
        } else {
            if (prev >= 0) minDiff = minOf(minDiff, curr.`val` - prev)
            prev = curr.`val`
            curr = curr.right
        }
    }
    return minDiff
}

```

