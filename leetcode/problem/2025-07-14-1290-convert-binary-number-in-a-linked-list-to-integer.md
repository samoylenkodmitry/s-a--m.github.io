---
layout: leetcode-entry
title: "1290. Convert Binary Number in a Linked List to Integer"
permalink: "/leetcode/problem/2025-07-14-1290-convert-binary-number-in-a-linked-list-to-integer/"
leetcode_ui: true
entry_slug: "2025-07-14-1290-convert-binary-number-in-a-linked-list-to-integer"
---

[1290. Convert Binary Number in a Linked List to Integer](https://leetcode.com/problems/convert-binary-number-in-a-linked-list-to-integer/description/) easy
[blog post](https://leetcode.com/problems/convert-binary-number-in-a-linked-list-to-integer/solutions/6956283/kotlin-rust-by-samoylenkodmitry-wky4/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/14072025-1290-convert-binary-number?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/3rOo8GcsOcM)

![1.webp](/assets/leetcode_daily_images/9addd802.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1049

#### Problem TLDR

Binary linked list to decimal #easy #linkedlist

#### Intuition

x = x * 2 + value

* use recursion
* use loop
* use values to hold some data

#### Approach

* try to write it in all difference ways

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 128ms
    fun ListNode?.str(): String = if (this == null) ""
        else "" + `val` + next.str()
    fun getDecimalValue(head: ListNode?) = head.str().toInt(2)

```
```kotlin

// 0ms
    fun getDecimalValue(head: ListNode?): Int =
        if (head?.next == null) head!!.`val` else
        getDecimalValue(head.next!!.apply { `val` += 2 * head.`val`})

```
```kotlin

// 0ms
    fun getDecimalValue(head: ListNode?, r: Int = 0): Int =
        head?.run { getDecimalValue(next, r * 2 + `val`) } ?: r

```
```kotlin

// 0ms
    fun getDecimalValue(head: ListNode?): Int {
        var x = head; var y = 0
        while (x != null) { y = y * 2 + x.`val`; x = x.next }
        return y
    }

```
```kotlin

// 0ms
    var max = 1
    fun getDecimalValue(head: ListNode?): Int = head?.run {
        val curr = max++
        getDecimalValue(next) + `val` * (1 shl (max - curr - 1))
    } ?: 0

```
```kotlin

// 0ms
    fun getDecimalValue(head: ListNode?): Int = head?.run {
        val curr = `val` / 2
        next?.`val` += (curr + 1) * 2
        val tail = getDecimalValue(next)
        val max = max(curr, (next?.`val` ?: 0) / 2)
        `val` = `val` % 2 + max * 2
        (`val` % 2) * (1 shl (max - curr)) + tail
    } ?: 0

```
```rust

// 0ms
    pub fn get_decimal_value(mut head: Option<Box<ListNode>>) -> i32 {
        let mut r = 0;
        while let Some(b) = head {
            r = r * 2 + b.val;
            head = b.next
        } r
    }

```
```c++

// 0ms
    int getDecimalValue(ListNode* head) {
        for (;; head = head->next)
            if (!head->next) return head->val;
            else head->next->val += head->val * 2;
    }

```

