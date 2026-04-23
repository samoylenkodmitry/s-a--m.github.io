---
layout: leetcode-entry
title: "234. Palindrome Linked List"
permalink: "/leetcode/problem/2024-03-22-234-palindrome-linked-list/"
leetcode_ui: true
entry_slug: "2024-03-22-234-palindrome-linked-list"
---

[234. Palindrome Linked List](https://leetcode.com/problems/palindrome-linked-list/description/) easy
[blog post](https://leetcode.com/problems/palindrome-linked-list/solutions/4909180/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/22032024-234-palindrome-linked-list?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/zj9qov2HWfE)
![2024-03-22_10-03.jpg](/assets/leetcode_daily_images/fd60cfbd.webp)

#### Problem TLDR

Is Linked List a palindrome #easy

#### Intuition

Find the middle using tortoise and hare algorithm and reverse it simultaneously.

#### Approach

* the corners case is to detect `odd` or `even` count of nodes and do the extra move
* gave up on the Rust solution without `clone()`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$, O(n) in Rust

#### Code

```kotlin

  fun isPalindrome(head: ListNode?): Boolean {
    var fast = head; var slow = head
    var prev: ListNode? = null
    while (fast?.next != null) {
      fast = fast?.next?.next
      val next = slow?.next
      slow?.next = prev
      prev = slow
      slow = next
    }
    if (fast != null) slow = slow?.next
    while (prev != null && prev?.`val` == slow?.`val`)
      prev = prev?.next.also { slow = slow?.next }
    return prev == null
  }

```
```rust

  pub fn is_palindrome(head: Option<Box<ListNode>>) -> bool {
    let (mut fast, mut slow, mut prev) = (head.clone(), head, None);
    while fast.is_some() && fast.as_ref().unwrap().next.is_some() {
        fast = fast.unwrap().next.unwrap().next;
        let mut slow_box = slow.unwrap();
        let next = slow_box.next;
        slow_box.next = prev;
        prev = Some(slow_box);
        slow = next
    }
    if fast.is_some() { slow = slow.unwrap().next }
    while let Some(prev_box) = prev {
      let slow_box = slow.unwrap();
      if prev_box.val != slow_box.val { return false }
      prev = prev_box.next; slow = slow_box.next
    }; true
  }

```

