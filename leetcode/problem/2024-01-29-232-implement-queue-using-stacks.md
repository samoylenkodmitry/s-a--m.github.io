---
layout: leetcode-entry
title: "232. Implement Queue using Stacks"
permalink: "/leetcode/problem/2024-01-29-232-implement-queue-using-stacks/"
leetcode_ui: true
entry_slug: "2024-01-29-232-implement-queue-using-stacks"
---

[232. Implement Queue using Stacks](https://leetcode.com/problems/implement-queue-using-stacks/description/) easy
[blog post](https://leetcode.com/problems/implement-queue-using-stacks/solutions/4641938/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/29012024-232-implement-queue-using?r=2bam17&utm_campaign=post&utm_medium=web&showWelcome=true)
[youtube](https://youtu.be/ZJnPxa6nRtw)
![image.png](/assets/leetcode_daily_images/12a86d02.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/487

#### Problem TLDR

Queue by 2 stacks.

#### Intuition

Let's write down how the numbers are added:
```
stack a: [1 2]
stack b: []

peek:

a: [1]
b: [2]

a: []
b: [2 1], b.peek == 1
```

#### Approach

Let's do some code golf.

#### Complexity

- Time complexity:
$$O(1)$$ for total operations. In general, stack drain is a rare operation

- Space complexity:
$$O(n)$$ for total operations.

#### Code

```kotlin

class MyQueue() {
  val a = Stack<Int>()
  val b = Stack<Int>()
  fun push(x: Int) = a.push(x)
  fun pop() = peek().also { b.pop() }
  fun peek(): Int {
    if (b.size < 1) while (a.size > 0) b += a.pop()
    return b.peek()
  }
  fun empty() = a.size + b.size == 0
}

```
```rust
struct MyQueue(Vec<i32>, Vec<i32>);
impl MyQueue {
    fn new() -> Self { Self(vec![], vec![]) }
    fn push(&mut self, x: i32) { self.0.push(x); }
    fn pop(&mut self) -> i32 { self.peek(); self.1.pop().unwrap() }
    fn peek(&mut self) -> i32 {
      if self.1.is_empty() { self.1.extend(self.0.drain(..).rev()); }
      *self.1.last().unwrap()
    }
    fn empty(&self) -> bool { self.0.len() + self.1.len() == 0 }
}
```

