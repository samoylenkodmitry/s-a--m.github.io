---
layout: leetcode-entry
title: "109. Convert Sorted List to Binary Search Tree"
permalink: "/leetcode/problem/2023-03-11-109-convert-sorted-list-to-binary-search-tree/"
leetcode_ui: true
entry_slug: "2023-03-11-109-convert-sorted-list-to-binary-search-tree"
---

[109. Convert Sorted List to Binary Search Tree](https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/description/) medium

[blog post](https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/solutions/3282962/kotlin-recursion/)

```kotlin

fun sortedListToBST(head: ListNode?): TreeNode? {
    if (head == null) return null
    if (head.next == null) return TreeNode(head.`val`)
    var one = head
    var twoPrev = head
    var two = head
    while (one != null && one.next != null) {
        one = one.next?.next
        twoPrev = two
        two = two?.next
    }
    twoPrev!!.next = null
    return TreeNode(two!!.`val`).apply {
        left = sortedListToBST(head)
        right = sortedListToBST(two!!.next)
    }
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/145
#### Intuition
One way is to convert linked list to array, then just build a binary search tree using divide and conquer technique. This will take $$O(nlog_2(n))$$ additional memory, and $$O(n)$$ time.
We can skip using the array and just compute the middle of the linked list each time.
#### Approach
Compute the middle of the linked list.
* careful with corner cases (check `fast.next != null` instead of `fast != null`)
#### Complexity
- Time complexity:
$$O(nlog_2(n))$$
- Space complexity:
$$O(log_2(n))$$ of additional space (for recursion)

