import type { FeedbackReport } from "@signsaarthi/shared";
import type { FormEvent } from "react";
import type { ReactElement } from "react";
import { useEffect, useRef } from "react";

type FeedbackReason = FeedbackReport["reason"];

type FeedbackDialogProps = {
  isOpen: boolean;
  isSubmitting: boolean;
  onClose: () => void;
  onSubmit: (input: { reason: FeedbackReason; comment: string }) => Promise<void>;
};

export function FeedbackDialog({
  isOpen,
  isSubmitting,
  onClose,
  onSubmit
}: FeedbackDialogProps): ReactElement | null {
  const dialogRef = useRef<HTMLFormElement>(null);

  useEffect(() => {
    if (!isOpen) {
      return undefined;
    }

    const previouslyFocused = document.activeElement instanceof HTMLElement ? document.activeElement : undefined;
    const getFocusableElements = (): HTMLElement[] =>
      Array.from(
        dialogRef.current?.querySelectorAll<HTMLElement>(
          'button, select, textarea, input, [href], [tabindex]:not([tabindex="-1"])'
        ) ?? []
      ).filter((element) => !element.hasAttribute("disabled"));

    window.setTimeout(() => {
      const preferredFocus = dialogRef.current?.querySelector<HTMLElement>("[data-autofocus]");
      preferredFocus?.focus();
      if (!preferredFocus) {
        getFocusableElements()[0]?.focus();
      }
    }, 0);

    function handleKeyDown(event: KeyboardEvent): void {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
        return;
      }
      if (event.key !== "Tab") {
        return;
      }

      const focusableElements = getFocusableElements();
      const firstElement = focusableElements[0];
      const lastElement = focusableElements.at(-1);
      if (!firstElement || !lastElement) {
        return;
      }
      if (event.shiftKey && document.activeElement === firstElement) {
        event.preventDefault();
        lastElement.focus();
      }
      if (!event.shiftKey && document.activeElement === lastElement) {
        event.preventDefault();
        firstElement.focus();
      }
    }

    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      previouslyFocused?.focus();
    };
  }, [isOpen, onClose]);

  if (!isOpen) {
    return null;
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    await onSubmit({
      reason: String(formData.get("reason") ?? "wrong_sign") as FeedbackReason,
      comment: String(formData.get("comment") ?? "")
    });
  }

  return (
    <div className="dialog-backdrop" role="presentation">
      <form
        ref={dialogRef}
        className="feedback-dialog"
        onSubmit={handleSubmit}
        role="dialog"
        aria-modal="true"
        aria-labelledby="feedback-dialog-title"
      >
        <div className="dialog-header">
          <h2 id="feedback-dialog-title">Report wrong sign</h2>
          <button type="button" className="icon-button" onClick={onClose} aria-label="Close feedback dialog">
            x
          </button>
        </div>
        <label className="field">
          Reason
          <select name="reason" defaultValue="wrong_sign" data-autofocus>
            <option value="wrong_sign">Wrong sign</option>
            <option value="incorrect_translation">Incorrect translation</option>
            <option value="missing_sign">Missing sign</option>
            <option value="poor_avatar_motion">Poor avatar movement</option>
            <option value="regional_variation">Regional variation</option>
            <option value="other">Other</option>
          </select>
        </label>
        <label className="field">
          Comment
          <textarea name="comment" rows={4} placeholder="Add context for the reviewer" />
        </label>
        <div className="dialog-actions">
          <button type="button" className="secondary-button" onClick={onClose}>
            Cancel
          </button>
          <button type="submit" className="primary-button" disabled={isSubmitting}>
            {isSubmitting ? "Submitting..." : "Submit feedback"}
          </button>
        </div>
      </form>
    </div>
  );
}
