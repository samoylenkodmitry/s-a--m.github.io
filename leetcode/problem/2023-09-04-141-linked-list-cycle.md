---
layout: leetcode-entry
title: "141. Linked List Cycle"
permalink: "/leetcode/problem/2023-09-04-141-linked-list-cycle/"
leetcode_ui: true
entry_slug: "2023-09-04-141-linked-list-cycle"
---

[141. Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/description/) easy
[blog post](https://leetcode.com/problems/linked-list-cycle/solutions/3999368/kotlin-one-liner/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/4092023-141-linked-list-cycle?utm_campaign=post&utm_medium=web)

![image.png](/assets/leetcode_daily_images/d5a0e81c.webp)

#### Problem TLDR

Detect a cycle in a `LinkedList`

#### Intuition

Use tortoise and rabbit technique

#### Approach

Move one pointer one step at a time, another two steps at a time. If there is a cycle, they will meet.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(log(n))$$ for recursion (iterative version is O(1))

#### Code

```kotlin

    fun hasCycle(slow: ListNode?, fast: ListNode? = slow?.next): Boolean =
      fast != null && (slow == fast || hasCycle(slow?.next, fast?.next?.next))

```

