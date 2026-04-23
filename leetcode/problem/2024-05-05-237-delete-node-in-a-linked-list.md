---
layout: leetcode-entry
title: "237. Delete Node in a Linked List"
permalink: "/leetcode/problem/2024-05-05-237-delete-node-in-a-linked-list/"
leetcode_ui: true
entry_slug: "2024-05-05-237-delete-node-in-a-linked-list"
---

[237. Delete Node in a Linked List](https://leetcode.com/problems/delete-node-in-a-linked-list/description/) medium
[blog post](https://leetcode.com/problems/delete-node-in-a-linked-list/solutions/5114307/kotlin-c/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/05052024-237-delete-node-in-a-linked?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/GZNHEMZEt3o)
![2024-05-05_08-14.webp](/assets/leetcode_daily_images/b8af3519.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/593

#### Problem TLDR

Delete current node in a Linked List #medium

#### Intuition

The O(n) solution is trivial: swap current and next values until the last node reached.
There is an O(1) solution exists, and it's clever: remove just the next node.

#### Approach

No Rust solution, as there is no template for it in leetcode.com.

#### Complexity

- Time complexity:
$$O(1)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun deleteNode(node: ListNode?) {
        node?.`val` = node?.next?.`val`
        node?.next = node?.next?.next
    }

```
```c++

    void deleteNode(ListNode* node) {
        *node = *node->next;
    }

```

